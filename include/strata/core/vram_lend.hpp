#pragma once
// VRAM Lend: lending a card's expert-cache memory to another program on the same machine, and taking it back.  The
// parts here know nothing of the cache: the control-line commands and their answers, the table of open leases and
// the rule for how much a card can lend.  The engine supplies only the LendCache interface over its expert cache
// stages (one per card on a layer split).
//
// The commands (one control line each; the words are case-sensitive, extra blanks and a trailing CR are ignored):
//
//   LEND <dev> <mib>    lend <mib> MiB of card <dev>, rounded UP to whole blocks of that card
//       -> LENT <id> <dev> <mib>    the lease, with the MiB actually lent
//       -> ERR MAX <mib>            more than the card can lend; <mib> is what it can lend now
//       -> ERR NODEV <dev>          <dev> is not a card that lends (well formed, no such card)
//   RETURN <id>         give a lease's memory back to the cache
//       -> RETURNED <id>
//       -> ERR BUSY <id>            someone still holds the memory: the lease stays open, try again later
//       -> ERR UNKNOWN <id>         no open lease has that id (never handed out, or already returned)
//   LENDABLE            what each card can lend now and what it has out
//       -> LENDABLE dev=<n> lendable=<mib> lent=<mib>    one line per card, in the engine's order
//       -> END
//
// A lend command that does not parse is answered, never passed on to the engine's other commands:
//       -> ERR MALFORMED LEND <dev> <mib>   <dev> not a non-negative int, or <mib> missing, zero, signed, not a
//                                           whole number or too large, or a word too many
//       -> ERR MALFORMED RETURN <id>        no id, or a word too many
//       -> ERR MALFORMED LENDABLE           a word after LENDABLE
//
// Lease ids are opaque to the caller (decimal numbers from 1, never 0) and unique for the life of the process, also
// across several LendCommands.  The engine does not record who borrows: a lease is an id, a card and a size.
//
// The commands come while requests run.  The engine's stdin reader thread passes each line through a LendQueue, and
// the engine's command loop takes them on its own thread: LENDABLE at any line, LEND and RETURN where nothing is in
// flight.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

/// The lendable rule: what a card can lend is its cached pool minus what the prompt path borrows from it minus the
/// minimum slots the cache keeps, clamped at 0 and rounded DOWN to whole blocks, so a lend never pushes the prompt
/// path back to reading token by token.  All in bytes; the result is a multiple of `block_bytes` (0 when
/// `block_bytes` is 0: a card with no lend blocks lends nothing).
///
/// `pool_bytes` is the pool as it is NOW, after the blocks already lent have left it: the rule does not subtract the
/// open leases itself, so each lease shrinks what is lendable by its own size and a returned one grows it again.
std::size_t lendable_bytes(std::size_t pool_bytes, std::size_t prompt_borrow_bytes, std::size_t min_slots_bytes,
                           std::size_t block_bytes);

/// One card that can lend, as the cache reports it: lendable_bytes() of its pool now, and the lend unit
/// (`block_bytes` 0: the card lends nothing).
struct LendCard {
    int dev = 0;
    std::size_t lendable_bytes = 0;
    std::size_t block_bytes = 0;
};

/// A lend's outcome.  ok: `bytes` taken out of the cache (at least what was asked, in whole blocks) and the cache's
/// own `token` for them, which comes back unchanged in give_back().  Refused: `max_bytes`, what the card can lend now.
struct LendTaken {
    bool ok = false;
    std::size_t bytes = 0;
    std::uint64_t token = 0;
    std::size_t max_bytes = 0;
};

enum class GiveBack {
    returned,   // the memory is back in the cache
    busy,       // the memory is still held by someone else: nothing changed, the caller tries again later
};

/// What the engine implements over its expert cache, per card.  The calls come from the engine's command loop, one
/// at a time, where nothing is in flight on any card (lend_cards() also between two tokens).
class LendCache {
public:
    virtual ~LendCache() = default;
    /// Every card that can lend, in a fixed order (a card with nothing lendable now is still listed).
    virtual std::vector<LendCard> lend_cards() = 0;
    /// Take `bytes` on card `dev` out of the cache.  LendCommands has already rounded `bytes` up to whole blocks and
    /// checked it against lend_cards(); the cache may still refuse (its pool changed) with the max it has now.
    virtual LendTaken lend(int dev, std::size_t bytes) = 0;
    /// Put back the memory of a lease lend() returned `token` for.  Busy leaves the lease open.
    virtual GiveBack give_back(int dev, std::uint64_t token) = 0;
};

/// A card's state as the LENDABLE lines show it: the cache's figures plus what the open leases hold, in whole MiB
/// (rounded down; the blocks are whole MiB, so nothing is lost).
struct LendCardStatus {
    int dev = 0;
    std::size_t lendable_mib = 0;
    std::size_t lent_mib = 0;
    std::size_t block_bytes = 0;
};

/// The segments of a card's cache a lease may take: the whole segments between the end of the minimum slots the cache
/// keeps (`floor_end`, a byte offset in its arena) and the start of the largest region the prompt path borrows for its
/// buffers (`loan_start`), so a lease never takes a floor slot and the prompt path's loan never meets a hole.  Segments
/// [first, end) of `seg_bytes` each: the lendable rule's pool - prompt borrow - minimum rounded down to whole segments,
/// or one segment fewer (a segment an edge falls inside stays out of the band).  Empty when the edges
/// meet or `seg_bytes` is 0.
struct LendBand {
    std::int64_t first = 0, end = 0;
    std::int64_t segments() const { return end > first ? end - first : 0; }
};
inline LendBand lend_band(std::uint64_t floor_end, std::uint64_t loan_start, std::uint64_t seg_bytes) {
    if (seg_bytes == 0) return {};
    return {(std::int64_t) ((floor_end + seg_bytes - 1) / seg_bytes), (std::int64_t) (loan_start / seg_bytes)};
}

/// LendCache over segmented expert caches, one per card (a layer split has one per stage); a segment is the lend
/// block.  `Cache` gives the segment layout: segmented(), segment_bytes(), segments(), segment_mapped(i), slots() and
/// slot_offset(s) (strata::core::ExpertCache).  A card's band is fixed when it is added, from the slots its floor keeps
/// and the most slots the prompt path's loan takes from the tail; what it can lend is the band's segments mapped now,
/// so a lease shrinks it by its own size and a return grows it again.  Should the live slots get fewer (#533's shrink
/// of the tail), the loan moves down with them and the band's end follows it.
///
/// No lease is taken here: lend() refuses every lend with nothing lendable now (ERR MAX 0) and give_back() has
/// nothing to give back.  The engine derives from it to take and return the segments.
template <class Cache>
class SegmentLendCache : public LendCache {
public:
    struct Card {
        int dev = 0;
        const Cache* cache = nullptr;
        std::int64_t floor_slots = 0;   // the leading slots the cache always keeps
        std::int64_t loan_slots = 0;    // the most slots the prompt path's loan takes from the tail (0: no loan)
        LendBand band;                  // fixed by add()
    };

    /// A card, in the order LENDABLE lists the cards.  A cache that is not segmented lends nothing (it is still
    /// listed).  The reference is good until the next add().
    const Card& add(int dev, const Cache& cache, std::int64_t floor_slots, std::int64_t loan_slots) {
        Card c;
        c.dev = dev;
        c.cache = &cache;
        c.floor_slots = floor_slots;
        c.loan_slots = loan_slots;
        if (cache.segmented()) {
            const std::int64_t n = cache.slots();
            const std::int64_t lo = floor_slots < 0 ? 0 : floor_slots > n ? n : floor_slots;
            c.band = lend_band(cache.slot_offset(lo), cache.slot_offset(loan_start(c)),
                               (std::uint64_t) cache.segment_bytes());
        }
        cards_.push_back(c);
        return cards_.back();
    }
    const std::vector<Card>& cards() const { return cards_; }

    /// The band's segments mapped now, up to where the prompt path's loan starts now.
    std::int64_t lendable_segments(const Card& c) const {
        const Cache& x = *c.cache;
        if (!x.segmented() || c.band.segments() == 0) return 0;
        std::int64_t end = (std::int64_t) (x.slot_offset(loan_start(c)) / (std::uint64_t) x.segment_bytes());
        if (end > c.band.end) end = c.band.end;
        if (end > x.segments()) end = x.segments();
        std::int64_t k = 0;
        for (std::int64_t i = c.band.first; i < end; ++i) k += x.segment_mapped(i) ? 1 : 0;
        return k;
    }

    std::vector<LendCard> lend_cards() override {
        std::vector<LendCard> out;
        out.reserve(cards_.size());
        for (const Card& c : cards_) {
            const std::size_t seg = c.cache->segmented() ? (std::size_t) c.cache->segment_bytes() : 0;
            out.push_back({c.dev, (std::size_t) lendable_segments(c) * seg, seg});
        }
        return out;
    }
    LendTaken lend(int, std::size_t) override { return {}; }
    GiveBack give_back(int, std::uint64_t) override { return GiveBack::busy; }

private:
    // the first slot of the prompt path's largest loan in the live slots now (the end of them without a loan)
    static std::int64_t loan_start(const Card& c) {
        const std::int64_t n = c.cache->slots();
        return c.loan_slots <= 0 ? n : c.loan_slots >= n ? 0 : n - c.loan_slots;
    }
    std::vector<Card> cards_;
};

/// What a control line is to the lend commands, by its first word (split on blanks as LendCommands::handle splits it):
/// not a lend command, LENDABLE (it only reads the cache's figures: answered at any line, also between two tokens of
/// a request), or LEND / RETURN (they change the cache: answered only where nothing is in flight on any card).  A
/// malformed lend command is still one of them: LendCommands answers it.
enum class LendLineKind { none, lendable, change };
LendLineKind lend_line_kind(const std::string& line);

/// The lend command lines waiting for the engine's command loop.  The stdin reader thread routes every line through
/// route(), which keeps the lend commands and leaves the rest to the engine's own queue; the command loop takes them in
/// the order they came.  waiting() is a lock-free look for the loop's hot paths (the decode polls it every token).
class LendQueue {
public:
    /// The reader: a lend command is queued (true); any other line is not (false), for the engine's own commands.
    bool route(const std::string& line);
    /// A lend command waits.
    bool waiting() const { return n_.load(std::memory_order_acquire) > 0; }
    /// The command loop: the next lend command, of either kind (false: none waits).
    bool take(std::string& line);
    /// The next one only while it is a LENDABLE: a LEND or RETURN in front keeps it, and those after it, in the queue
    /// (the answers go out in the order the commands came).
    bool take_lendable(std::string& line);

private:
    std::mutex mu_;
    std::deque<std::string> q_;
    std::atomic<std::size_t> n_{0};
};

/// The lend commands over one engine: holds the open leases and turns a control line into its answer.  Keep one for
/// the life of the engine (the leases live in it).  Not thread-safe: call it from the engine's command loop.
class LendCommands {
public:
    explicit LendCommands(LendCache& cache) : cache_(cache) {}

    /// A lend command (LEND, RETURN, LENDABLE): true, and `reply` holds its answer, one or more lines each ending in
    /// '\n', for the loop to print as it is (std::fputs, then std::fflush).  Anything else: false and `reply` empty,
    /// for the loop's own commands.
    bool handle(const std::string& line, std::string& reply);

    /// Every card's figures, as LENDABLE reports them.
    std::vector<LendCardStatus> status();
    std::size_t open_leases() const { return leases_.size(); }

private:
    struct Lease {
        int dev = 0;
        std::size_t bytes = 0;
        std::uint64_t token = 0;
    };
    std::size_t lent_bytes(int dev) const;

    LendCache& cache_;
    std::map<std::uint64_t, Lease> leases_;   // by lease id
};

}  // namespace strata::core

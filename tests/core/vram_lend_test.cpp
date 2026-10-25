// VRAM Lend: the lend commands, the lease table and the lendable rule, against a fake expert cache - no GPU, no CUDA.
// Every row of the LEND / RETURN / LENDABLE answer tables of the lend protocol, extra blanks and a trailing CR, lease
// ids from 1 that are never handed out twice, and the queue between the stdin reader and the command loop.
#include "strata/core/vram_lend.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

using strata::core::GiveBack;
using strata::core::LendCache;
using strata::core::LendCard;
using strata::core::LendCommands;
using strata::core::LendLineKind;
using strata::core::LendQueue;
using strata::core::LendTaken;
using strata::core::SegmentLendCache;
using strata::core::lend_band;
using strata::core::lend_line_kind;
using strata::core::lendable_bytes;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)
#define CHECK_REPLY(desk, line, want) do { std::string r_; const std::string l_ = (line), w_ = (want); \
    if (!(desk).handle(l_, r_) || r_ != w_) { \
        std::fprintf(stderr, "FAIL line %d: '%s' gave '%s', want '%s'\n", __LINE__, l_.c_str(), r_.c_str(), \
                     w_.c_str()); return 1; } \
    } while (0)

namespace {

constexpr size_t M = size_t{1} << 20;

// One card of the fake cache: a pool of whole blocks (segments), the prompt path's borrow and the minimum slots it
// keeps.  The blocks lent leave the pool, as they leave a real cache, so the rule sees the pool after lends.
struct FakeCard {
    int dev;
    size_t pool, borrow, min_slots, block;   // bytes
};

struct FakeCache : LendCache {
    std::vector<FakeCard> cards;
    std::map<uint64_t, std::pair<int, size_t>> leases;   // token -> (dev, bytes)
    std::set<uint64_t> busy;                             // tokens whose memory someone still holds
    uint64_t next_token = 1000;
    size_t refuse_max = SIZE_MAX;   // below SIZE_MAX: the next lend is refused with this max (a pool that shrank)
    std::vector<size_t> asked;      // the bytes of every lend() call

    FakeCard* card(int dev) {
        for (FakeCard& c : cards)
            if (c.dev == dev) return &c;
        return nullptr;
    }
    std::vector<LendCard> lend_cards() override {
        std::vector<LendCard> out;
        for (const FakeCard& c : cards) out.push_back({c.dev, lendable_bytes(c.pool, c.borrow, c.min_slots, c.block), c.block});
        return out;
    }
    LendTaken lend(int dev, size_t bytes) override {
        asked.push_back(bytes);
        LendTaken r;
        FakeCard* c = card(dev);
        if (!c) return r;
        if (refuse_max != SIZE_MAX) {
            r.max_bytes = refuse_max;
            refuse_max = SIZE_MAX;
            return r;
        }
        c->pool -= bytes;
        r.ok = true;
        r.bytes = bytes;
        r.token = next_token++;
        leases[r.token] = {dev, bytes};
        return r;
    }
    GiveBack give_back(int dev, uint64_t token) override {
        if (busy.count(token)) return GiveBack::busy;
        auto it = leases.find(token);
        if (it == leases.end() || it->second.first != dev) return GiveBack::busy;   // the table never asks this
        card(dev)->pool += it->second.second;
        leases.erase(it);
        return GiveBack::returned;
    }
};

// "LENT <id> <dev> <mib>\n" -> id (0 when the reply is not that LENT line)
uint64_t lent_id(const std::string& reply, int dev, size_t mib) {
    unsigned long long id = 0;
    int d = -1;
    size_t m = 0;
    char tail = 0;
    if (std::sscanf(reply.c_str(), "LENT %llu %d %zu%c", &id, &d, &m, &tail) != 4 || tail != '\n') return 0;
    if (reply != "LENT " + std::to_string(id) + " " + std::to_string(d) + " " + std::to_string(m) + "\n") return 0;
    return d == dev && m == mib ? (uint64_t) id : 0;
}

// A segmented expert cache as SegmentLendCache sees it: `n` live slots of `stride` bytes each, back to back, in
// segments of `seg` bytes (the last one shorter), each mapped or not.
struct FakeSegCache {
    int64_t n = 0;
    uint64_t stride = 0;
    int64_t seg = 0;   // 0: not segmented
    std::vector<bool> mapped;

    FakeSegCache(int64_t slots, uint64_t slot_bytes, int64_t seg_bytes) : n(slots), stride(slot_bytes), seg(seg_bytes) {
        if (seg > 0) mapped.assign((size_t) ((n * stride + seg - 1) / seg), true);
    }
    bool segmented() const { return seg > 0; }
    int64_t segment_bytes() const { return seg; }
    int64_t segments() const { return (int64_t) mapped.size(); }
    bool segment_mapped(int64_t i) const { return i >= 0 && i < segments() && mapped[(size_t) i]; }
    int64_t slots() const { return n; }
    uint64_t slot_offset(int64_t s) const { return (uint64_t) s * stride; }
    // the slots [lo, hi) that touch segment i (as ExpertCache::segment_slots: a slot that straddles its edge counts)
    void segment_slots(int64_t i, int64_t& lo, int64_t& hi) const {
        const uint64_t a = (uint64_t) i * (uint64_t) seg, b = a + (uint64_t) seg;
        lo = (int64_t) (a / stride);
        hi = (int64_t) ((b + stride - 1) / stride);
        if (lo > n) lo = n;
        if (hi > n) hi = n;
    }
};

}  // namespace

int main() {
    // ---- the lendable rule: pool - prompt borrow - minimum slots, clamped at 0, down to whole blocks (bytes)
    CHECK(lendable_bytes(640 * M, 64 * M, 128 * M, 64 * M) == 448 * M);
    CHECK(lendable_bytes(1000 * M, 250 * M, 300 * M, 100 * M) == 400 * M);   // 450 MiB -> 4 blocks of 100
    CHECK(lendable_bytes(1000 * M, 250 * M, 300 * M, 100 * M - 1) == 4 * (100 * M - 1));
    CHECK(lendable_bytes(10, 0, 0, 3) == 9);
    CHECK(lendable_bytes(192 * M, 64 * M, 128 * M, 64 * M) == 0);   // exactly used up
    CHECK(lendable_bytes(100 * M, 64 * M, 128 * M, 64 * M) == 0);   // short: 0, not a wrapped huge value
    CHECK(lendable_bytes(100 * M, 200 * M, 0, 64 * M) == 0);
    CHECK(lendable_bytes(100 * M, 0, 200 * M, 64 * M) == 0);
    CHECK(lendable_bytes(SIZE_MAX, SIZE_MAX, SIZE_MAX, 64 * M) == 0);
    CHECK(lendable_bytes(255 * M, 64 * M, 128 * M, 64 * M) == 0);   // 63 MiB left: less than one block
    CHECK(lendable_bytes(640 * M, 64 * M, 128 * M, 0) == 0);        // no block size (no lending there): nothing
    // the cache passes the pool after lends: 3 blocks lent from the 640 MiB pool leave 256 MiB lendable
    CHECK(lendable_bytes(640 * M - 192 * M, 64 * M, 128 * M, 64 * M) == 256 * M);

    FakeCache eng;
    eng.cards = {{0, 640 * M, 64 * M, 128 * M, 64 * M},      // lendable 448 MiB, blocks of 64
                 {2, 1000 * M, 250 * M, 300 * M, 100 * M},   // lendable 400 MiB, blocks of 100
                 {3, 100 * M, 64 * M, 128 * M, 64 * M}};     // lendable 0: still listed
    LendCommands desk(eng);

    // ---- LENDABLE: one line per card in the cache's order, then END; extra blanks and a trailing CR are ignored
    const std::string idle = "LENDABLE dev=0 lendable=448 lent=0\n"
                             "LENDABLE dev=2 lendable=400 lent=0\n"
                             "LENDABLE dev=3 lendable=0 lent=0\n"
                             "END\n";
    CHECK_REPLY(desk, "LENDABLE", idle);
    CHECK_REPLY(desk, "  LENDABLE \r", idle);
    CHECK_REPLY(desk, "LENDABLE\r", idle);
    CHECK_REPLY(desk, "\tLENDABLE\t", idle);

    // ---- LEND: the first lease of the process is id 1; rounded up to whole blocks (100 -> 2 blocks of 64)
    std::string r;
    CHECK(desk.handle("LEND 0 100", r));
    const uint64_t a = lent_id(r, 0, 128);
    CHECK(a == 1);
    CHECK(!eng.asked.empty() && eng.asked.back() == 128 * M);   // the cache is asked for whole blocks, in bytes
    CHECK(desk.handle("  LEND   0  64 \r", r));                 // exactly one block: no rounding; blanks and CR
    const uint64_t b = lent_id(r, 0, 64);
    CHECK(b == 2);
    CHECK(desk.open_leases() == 2);
    CHECK(eng.leases.size() == 2);
    // lendable shrinks by what is out (448 - 192), lent shows it
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=256 lent=192\n"
                                  "LENDABLE dev=2 lendable=400 lent=0\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");

    // ---- ERR MAX: more than the card's lendable, checked before the round-up, with the lendable it has now
    const size_t calls = eng.asked.size();
    CHECK_REPLY(desk, "LEND 0 1000", "ERR MAX 256\n");
    CHECK_REPLY(desk, "LEND 0 257", "ERR MAX 256\n");
    CHECK_REPLY(desk, "LEND 2 401", "ERR MAX 400\n");
    CHECK_REPLY(desk, "LEND 3 1", "ERR MAX 0\n");
    CHECK_REPLY(desk, "LEND 0 18446744073709551615", "ERR MAX 256\n");   // the largest u64: no wrap in MiB to bytes
    CHECK_REPLY(desk, "LEND 0 17592186044416", "ERR MAX 256\n");         // 2^44 MiB: 2^64 bytes would wrap to 0
    CHECK(eng.asked.size() == calls);   // refused by the table: the cache is not asked
    CHECK(desk.open_leases() == 2);
    // the cache may still refuse (its pool shrank since it reported): its max is the answer, no lease, no id
    eng.refuse_max = 64 * M;
    CHECK_REPLY(desk, "LEND 0 128", "ERR MAX 64\n");
    CHECK(desk.open_leases() == 2);

    // ---- RETURN: two leases on one card come back independently, in either order
    const std::string ra = std::to_string(a), rb = std::to_string(b);
    eng.busy.insert(eng.leases.rbegin()->first);   // b's token (the later one): its memory is still held
    CHECK_REPLY(desk, "RETURN " + rb, "ERR BUSY " + rb + "\n");
    CHECK(desk.open_leases() == 2);   // a busy lease stays open, nothing changed
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=256 lent=192\n"
                                  "LENDABLE dev=2 lendable=400 lent=0\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");
    CHECK_REPLY(desk, " RETURN  " + ra + " \r", "RETURNED " + ra + "\n");
    CHECK(desk.open_leases() == 1);
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=384 lent=64\n"
                                  "LENDABLE dev=2 lendable=400 lent=0\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");
    CHECK_REPLY(desk, "RETURN " + ra, "ERR UNKNOWN " + ra + "\n");   // already returned
    CHECK_REPLY(desk, "RETURN " + rb, "ERR BUSY " + rb + "\n");      // still held: retry
    eng.busy.clear();
    CHECK_REPLY(desk, "RETURN " + rb, "RETURNED " + rb + "\n");      // later: the same id
    CHECK(desk.open_leases() == 0 && eng.leases.empty());
    CHECK_REPLY(desk, "RETURN " + rb, "ERR UNKNOWN " + rb + "\n");
    CHECK_REPLY(desk, "LENDABLE", idle);   // everything back: the figures of before the lends

    // two leases on another card, the later one returned first; the ids go on from 3 (the refused lends took none,
    // the returned ones are not handed out again)
    CHECK(desk.handle("LEND 2 150", r));   // 2 blocks of 100
    const uint64_t c = lent_id(r, 2, 200);
    CHECK(desk.handle("LEND 2 1", r));     // 1 block
    const uint64_t d = lent_id(r, 2, 100);
    CHECK(c == 3 && d == 4);
    CHECK_REPLY(desk, "LEND 2 101", "ERR MAX 100\n");
    CHECK_REPLY(desk, "RETURN " + std::to_string(d), "RETURNED " + std::to_string(d) + "\n");
    CHECK_REPLY(desk, "LENDABLE", "LENDABLE dev=0 lendable=448 lent=0\n"
                                  "LENDABLE dev=2 lendable=200 lent=200\n"
                                  "LENDABLE dev=3 lendable=0 lent=0\n"
                                  "END\n");
    CHECK_REPLY(desk, "RETURN " + std::to_string(c), "RETURNED " + std::to_string(c) + "\n");
    CHECK(desk.handle("LEND 2 400", r));   // the whole lendable
    const uint64_t f = lent_id(r, 2, 400);
    CHECK(f == 5);
    CHECK_REPLY(desk, "LEND 2 1", "ERR MAX 0\n");

    // ---- ERR UNKNOWN: ids never handed out (the ids are opaque: any one word is looked up and echoed)
    CHECK_REPLY(desk, "RETURN 999999999", "ERR UNKNOWN 999999999\n");
    CHECK_REPLY(desk, "RETURN 0", "ERR UNKNOWN 0\n");
    CHECK_REPLY(desk, "RETURN abc", "ERR UNKNOWN abc\n");
    CHECK_REPLY(desk, "RETURN 99999999999999999999999", "ERR UNKNOWN 99999999999999999999999\n");

    // ---- ERR NODEV: well formed, but no such card
    CHECK_REPLY(desk, "LEND 5 10", "ERR NODEV 5\n");
    CHECK_REPLY(desk, "LEND 1 10", "ERR NODEV 1\n");
    CHECK_REPLY(desk, "LEND 007 10", "ERR NODEV 7\n");

    // ---- ERR MALFORMED: answered with an ERR, never passed on
    for (const char* bad : {"LEND", "LEND 0", "LEND x 10", "LEND -1 10", "LEND +1 10", "LEND 1.0 10", "LEND 0 0",
                            "LEND 0 00", "LEND 0 -5", "LEND 0 +5", "LEND 0 1.5", "LEND 0 1e3", "LEND 0 0x10",
                            "LEND 0 10MiB", "LEND 0 10 extra", "LEND 0 99999999999999999999999",
                            "LEND 99999999999 10", "  LEND  0 \r", "LEND 0 10 \r x"})
        CHECK_REPLY(desk, bad, "ERR MALFORMED LEND <dev> <mib>\n");
    for (const char* bad : {"RETURN", "RETURN 1 2", " RETURN \r", "RETURN 5 \r x"})
        CHECK_REPLY(desk, bad, "ERR MALFORMED RETURN <id>\n");
    for (const char* bad : {"LENDABLE 0", "LENDABLE x\r", " LENDABLE  dev=0 "})
        CHECK_REPLY(desk, bad, "ERR MALFORMED LENDABLE\n");
    CHECK(eng.leases.size() == 1 && desk.open_leases() == 1);   // none of them touched a lease

    // ---- not a lend command: false and no reply, for the engine's own loop to handle (the words are case-sensitive)
    for (const char* other : {"", "   ", "\r", "GEN 1 2 3", "QUIT", "STOP", "VRAM 4096 0", "VRAM - 1", "LENDX 0 10",
                              "lend 0 10", "Lend 0 10", "return 1", "lendable", "RETURNED 1", "LENDABLES", "LENT 1 0 64",
                              "END", "ERR MAX 0", "X LEND 0 64"}) {
        std::string o = "stale";
        CHECK(!desk.handle(other, o));
        CHECK(o.empty());
    }

    // ---- ids stay unique for the process life, also across another command table
    FakeCache eng2;
    eng2.cards = {{0, 640 * M, 0, 0, 64 * M}};
    LendCommands desk2(eng2);
    CHECK(desk2.handle("LEND 0 64", r));
    const uint64_t e = lent_id(r, 0, 64);
    CHECK(e == 6);
    CHECK_REPLY(desk, "RETURN " + std::to_string(e), "ERR UNKNOWN " + std::to_string(e) + "\n");
    CHECK_REPLY(desk2, "RETURN " + std::to_string(f), "ERR UNKNOWN " + std::to_string(f) + "\n");
    CHECK_REPLY(desk2, "RETURN " + std::to_string(e), "RETURNED " + std::to_string(e) + "\n");
    CHECK(desk2.handle("LEND 0 64", r) && lent_id(r, 0, 64) == 7);

    // a cache with no card that lends: LENDABLE is END alone, every LEND is NODEV
    FakeCache none;
    LendCommands desk3(none);
    CHECK_REPLY(desk3, "LENDABLE", "END\n");
    CHECK_REPLY(desk3, "LEND 0 64", "ERR NODEV 0\n");

    // status(): the same figures the LENDABLE lines show, with the block size
    const auto st = desk.status();
    CHECK(st.size() == 3 && st[1].dev == 2 && st[1].lendable_mib == 0 && st[1].lent_mib == 400 &&
          st[1].block_bytes == 100 * M);

    // ---- the control-line routing: which lines are lend commands, and which of them only read
    for (const char* l : {"LENDABLE", "  LENDABLE \r", "LENDABLE 0", "LENDABLE\t"})
        CHECK(lend_line_kind(l) == LendLineKind::lendable);
    for (const char* l : {"LEND 0 64", "LEND", "LEND x 10", " RETURN 7", "RETURN", "RETURN 1 2", "LEND 0 64\r"})
        CHECK(lend_line_kind(l) == LendLineKind::change);
    for (const char* l : {"", "   ", "GEN 1 2 3", "QUIT", "STOP", "STOP 3", "VRAM 4096 0", "VRAM - 1", "LENDX 0 10",
                          "lend 0 10", "RETURNED 1", "LENDABLES", "LENT 1 0 64", "END", "X LEND 0 64"})
        CHECK(lend_line_kind(l) == LendLineKind::none);

    // ---- the queue between the reader thread and the command loop: lend lines only, in order; a LENDABLE is taken
    // ahead of the loop's quiet point only while no LEND or RETURN is in front of it
    LendQueue q;
    std::string l;
    CHECK(!q.waiting() && !q.take(l) && !q.take_lendable(l));
    CHECK(!q.route("GEN 4 rid=1 1,2,3") && !q.route("STOP") && !q.route("") && !q.route("VRAM - 0") && !q.waiting());
    CHECK(q.route("LENDABLE"));
    CHECK(q.waiting());
    CHECK(q.route("LEND 0 64") && q.route("LENDABLE") && q.route("RETURN 3"));
    CHECK(q.take_lendable(l) && l == "LENDABLE");
    CHECK(!q.take_lendable(l) && q.waiting());   // a LEND in front: the LENDABLE behind it waits too
    CHECK(q.take(l) && l == "LEND 0 64");
    CHECK(q.take_lendable(l) && l == "LENDABLE");
    CHECK(!q.take_lendable(l) && q.waiting());
    CHECK(q.take(l) && l == "RETURN 3");
    CHECK(!q.waiting() && !q.take(l) && !q.take_lendable(l));
    // the reader thread and the loop at once: every lend line arrives once and in order, the others are left out
    {
        LendQueue tq;
        constexpr int kN = 2000;
        std::thread reader([&tq] {
            for (int i = 0; i < kN; ++i) {
                tq.route("GEN 1 rid=" + std::to_string(i) + " 1");
                tq.route(i % 3 == 0 ? "LENDABLE" : "RETURN " + std::to_string(i));
            }
        });
        int next = 0, polls = 0;
        while (next < kN && polls < 100000000) {
            ++polls;
            std::string got;
            if (!tq.take(got)) continue;
            const std::string want = next % 3 == 0 ? "LENDABLE" : "RETURN " + std::to_string(next);
            if (got != want) {
                std::fprintf(stderr, "FAIL queue order: got '%s', want '%s'\n", got.c_str(), want.c_str());
                reader.join();
                return 1;
            }
            ++next;
        }
        reader.join();
        CHECK(next == kN && !tq.waiting());
    }

    // ---- the lend band: the whole segments between the floor's end and the prompt loan's start (byte offsets)
    CHECK(lend_band(0, 1024 * M, 512 * M).first == 0 && lend_band(0, 1024 * M, 512 * M).end == 2);
    CHECK(lend_band(1, 1024 * M, 512 * M).first == 1 && lend_band(1, 1024 * M, 512 * M).segments() == 1);
    CHECK(lend_band(512 * M, 1024 * M - 1, 512 * M).segments() == 0);   // the loan starts inside segment 1
    CHECK(lend_band(300 * M, 1600 * M, 512 * M).first == 1 && lend_band(300 * M, 1600 * M, 512 * M).end == 3);
    CHECK(lend_band(1100 * M, 1000 * M, 512 * M).segments() == 0);      // the edges cross: nothing
    CHECK(lend_band(0, 1024 * M, 0).segments() == 0);                   // not segmented: nothing

    // ---- SegmentLendCache over the served layout's shape: 7551 slots of ~2.5 MiB, 512 MiB segments, the 128-slot
    // floor and a 1766-slot prompt loan
    {
        const uint64_t stride = 2600000;
        FakeSegCache c0(7551, stride, 512 * (int64_t) M), c1(6936, stride, 512 * (int64_t) M), flat(4000, stride, 0);
        SegmentLendCache<FakeSegCache> lc;
        const auto k0 = lc.add(0, c0, 128, 1766);   // (a copy: the next add moves the cards)
        CHECK(k0.band.first == 1);                                   // slot 127 ends inside segment 0
        CHECK(k0.band.end == (int64_t) ((7551 - 1766) * stride / (512 * M)));
        // within one segment of the rule: pool - prompt borrow - floor, in whole segments
        const size_t rule0 = lendable_bytes(7551 * stride, 1766 * stride, 128 * stride, 512 * M);
        const size_t band0 = (size_t) k0.band.segments() * 512 * M;
        CHECK(band0 <= rule0 && rule0 - band0 <= 512 * M);
        lc.add(1, c1, 128, 1766);
        lc.add(3, flat, 128, 1766);                                  // not segmented: listed, lends nothing
        std::vector<LendCard> v = lc.lend_cards();
        CHECK(v.size() == 3 && v[0].dev == 0 && v[1].dev == 1 && v[2].dev == 3);
        CHECK(v[0].block_bytes == 512 * M && v[0].lendable_bytes == band0);
        CHECK(v[1].lendable_bytes == (size_t) lc.cards()[1].band.segments() * 512 * M && v[1].lendable_bytes > 0);
        CHECK(v[2].block_bytes == 0 && v[2].lendable_bytes == 0);
        // the band's segments that are not mapped are not lendable; segments outside the band never count
        c0.mapped[(size_t) k0.band.first] = false;
        c0.mapped[(size_t) k0.band.first + 4] = false;
        c0.mapped[0] = false;
        c0.mapped[(size_t) k0.band.end] = false;
        CHECK(lc.lend_cards()[0].lendable_bytes == band0 - 2 * 512 * M);
        c0.mapped.assign(c0.mapped.size(), true);
        // fewer live slots (a shrink of the tail): the loan moves down and the band's end with it
        c0.n = 6000;
        CHECK(lc.lend_cards()[0].lendable_bytes ==
              (size_t) ((int64_t) ((6000 - 1766) * stride / (512 * M)) - k0.band.first) * 512 * M);
        c0.n = 7551;
        CHECK(lc.lend_cards()[0].lendable_bytes == band0);
        // no loan: the band reaches the last whole segment of the live slots; a loan of every slot: no band
        FakeSegCache c2(7551, stride, 512 * (int64_t) M);
        SegmentLendCache<FakeSegCache> lc2;
        CHECK(lc2.add(0, c2, 128, 0).band.end == (int64_t) (7551 * stride / (512 * M)));
        CHECK(lc2.add(1, c2, 128, 7551).band.segments() == 0);
        CHECK(lc2.add(2, c2, 8000, 0).band.segments() == 0);

        // LENDABLE through the lend commands; a LEND is refused with nothing lendable now (leases are not taken yet)
        LendCommands d(lc);
        const std::string want = "LENDABLE dev=0 lendable=" + std::to_string(band0 / M) + " lent=0\n" +
                                 "LENDABLE dev=1 lendable=" + std::to_string(v[1].lendable_bytes / M) + " lent=0\n" +
                                 "LENDABLE dev=3 lendable=0 lent=0\nEND\n";
        CHECK_REPLY(d, "LENDABLE", want);
        CHECK_REPLY(d, "LEND 0 512", "ERR MAX 0\n");
        CHECK_REPLY(d, "LEND 0 99999999", "ERR MAX " + std::to_string(band0 / M) + "\n");
        CHECK_REPLY(d, "LEND 3 1", "ERR MAX 0\n");
        CHECK_REPLY(d, "LEND 2 1", "ERR NODEV 2\n");
        CHECK_REPLY(d, "RETURN 1", "ERR UNKNOWN 1\n");
        CHECK(d.open_leases() == 0);
        CHECK_REPLY(d, "LENDABLE", want);
    }

    // ---- a lease takes the coldest whole segments of the band by the expert profile's rank, through the engine's
    // take hook, and the cache keeps which segments each lease holds
    {
        using SLC = SegmentLendCache<FakeSegCache>;
        using Cold = std::vector<std::pair<int64_t, double>>;
        // 100 slots of 10 MiB, 64 MiB segments (16, the last one shorter); a 10-slot floor and a 20-slot loan: the
        // band is segments [2, 12) (the floor ends at 100 MiB inside segment 1, the loan starts at 800 MiB)
        FakeSegCache c(100, 10 * M, 64 * (int64_t) M);
        SLC lc;
        const SLC::Card k = lc.add(0, c, 10, 20);
        CHECK(k.band.first == 2 && k.band.end == 12);
        CHECK(SLC::loan_start(k) == 80);
        std::vector<int64_t> ranks(100);
        for (int64_t s = 0; s < 100; ++s) ranks[(size_t) s] = s;   // as the profile fill leaves it: slot order is rank order
        // coldness per band segment: the mean rank of the slots that touch it, straddlers included
        const Cold cold = lc.band_coldness(k, ranks);
        CHECK(cold.size() == 10 && cold.front().first == 2 && cold.back().first == 11);
        int64_t lo = 0, hi = 0;
        c.segment_slots(2, lo, hi);
        CHECK(lo == 12 && hi == 20);   // 128..192 MiB: slots 12..19 (slot 19 straddles into segment 3)
        CHECK(cold.front().second == (12 + 19) / 2.0);
        // with ranks in slot order the coldest are the band's last segments, given in segment order
        CHECK((SLC::coldest(cold, 3) == std::vector<int64_t>{9, 10, 11}));
        CHECK(SLC::coldest(cold, 0).empty() && SLC::coldest(cold, 99).size() == 10);
        // the adaptive tier moved experts: cold ones sit in segment 4's slots, hot ones in segment 11's (neither in a
        // slot that straddles into a neighbour)
        std::vector<int64_t> moved = ranks;
        c.segment_slots(4, lo, hi);
        for (int64_t s = lo + 1; s < hi - 1; ++s) moved[(size_t) s] = 1000 + s;
        c.segment_slots(11, lo, hi);
        for (int64_t s = lo + 1; s < hi - 1; ++s) moved[(size_t) s] = s - 100;
        CHECK((SLC::coldest(lc.band_coldness(k, moved), 2) == std::vector<int64_t>{4, 10}));
        // a tie goes to the higher segment (nearer the tail)
        CHECK((SLC::coldest({{3, 5.0}, {7, 5.0}, {5, 5.0}}, 2) == std::vector<int64_t>{5, 7}));
        // a segment of empty slots (the engine ranks them one past the coldest expert) is the coldest
        std::vector<int64_t> empty = ranks;
        c.segment_slots(3, lo, hi);
        for (int64_t s = lo; s < hi; ++s) empty[(size_t) s] = 100;
        CHECK((SLC::coldest(lc.band_coldness(k, empty), 1) == std::vector<int64_t>{3}));

        // LEND through the lend commands.  No take hook: nothing can be taken (ERR MAX 0)
        LendCommands d(lc);
        CHECK_REPLY(d, "LEND 0 64", "ERR MAX 0\n");
        // the engine's hook gets the card, the chosen segments and the band's coldness now; it unmaps them (here: marks
        // them) and records what it evicted in the lease
        std::vector<std::vector<int64_t>> took;
        std::vector<size_t> cold_seen;
        bool refuse = false;
        int ranks_dev = -1;
        lc.set_ranks([&](const SLC::Card& cc) { ranks_dev = cc.dev; return moved; });
        lc.set_take([&](const SLC::Card& cc, strata::core::SegmentLease& l, const Cold& cl) {
            cold_seen.push_back(cl.size());
            if (refuse || cc.dev != 0) return false;
            took.push_back(l.segs);
            for (int64_t s : l.segs) c.mapped[(size_t) s] = false;
            l.evicted.push_back({7, (int32_t) l.segs[0]});
            return true;
        });
        // 100 MiB is two segments (rounded up): the two coldest, 4 and 10
        const size_t before = lc.lend_cards()[0].lendable_bytes;
        CHECK(before == 10 * 64 * M);
        std::string r;
        CHECK(d.handle("LEND 0 100", r) && lent_id(r, 0, 128) != 0);
        const uint64_t id1 = lent_id(r, 0, 128);
        CHECK(ranks_dev == 0 && took.size() == 1 && (took[0] == std::vector<int64_t>{4, 10}) && cold_seen[0] == 10);
        CHECK(lc.lend_cards()[0].lendable_bytes == before - 2 * 64 * M);
        CHECK(lc.leases().size() == 1 && lc.leases().begin()->second.dev == 0 &&
              (lc.leases().begin()->second.segs == std::vector<int64_t>{4, 10}) &&
              lc.leases().begin()->second.evicted.size() == 1);
        // the next lease takes the coldest of what is still mapped (segment 9), never a lent one
        CHECK(d.handle("LEND 0 64", r) && lent_id(r, 0, 64) != 0 && lent_id(r, 0, 64) != id1);
        CHECK(took.size() == 2 && (took[1] == std::vector<int64_t>{9}) && cold_seen[1] == 8);
        CHECK(lc.leases().size() == 2 && d.open_leases() == 2);
        CHECK_REPLY(d, "LENDABLE", "LENDABLE dev=0 lendable=448 lent=192\nEND\n");
        // more than is lendable: refused before the hook; a hook that fails: refused with what is lendable, no lease
        CHECK_REPLY(d, "LEND 0 449", "ERR MAX 448\n");
        CHECK(cold_seen.size() == 2);
        refuse = true;
        CHECK_REPLY(d, "LEND 0 64", "ERR MAX 448\n");
        CHECK(cold_seen.size() == 3 && took.size() == 2 && lc.leases().size() == 2 && d.open_leases() == 2);
        // a card that is not segmented never reaches the hook
        FakeSegCache flat(100, 10 * M, 0);
        SLC lc3;
        lc3.add(5, flat, 10, 20);
        bool called = false;
        lc3.set_take([&](const SLC::Card&, strata::core::SegmentLease&, const Cold&) { called = true; return true; });
        LendCommands d3(lc3);
        CHECK_REPLY(d3, "LEND 5 1", "ERR MAX 0\n");
        CHECK(!called);
    }

    std::printf("vram_lend_test: all checks passed\n");
    return 0;
}

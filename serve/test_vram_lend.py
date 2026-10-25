"""serve/test_vram_lend.py - the /vram routes (serve/vram_lend.py): lend a GPU's expert-cache memory to another program,
take it back and say what is lent, over a stand-in engine that speaks the lend commands (LEND / RETURN / LENDABLE) the
way the engine does - also while a request runs.  The contract is VRAM Lend's api.md and engine-protocol.md.

    python -m unittest serve.test_vram_lend -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, engine_args, serve  # noqa: E402
from serve import gpu_map, runconfig, vram_lend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The stand-in engine: three GPU-PCI lines (GPU 2 does not lend; only with --vram-lend, as the engine prints them),
# READY, then GEN (ids + 1000, one token every 30 ms, on a thread: the command loop reads on meanwhile), STOP, QUIT and
# the lend commands with the engine's answers - only when it was started with --vram-lend (without it, the generic ERR
# for a line it does not know).  GPUs lend in 64 MiB
# blocks (a lend is rounded up).  FAKE_LEND=0: an engine that refuses the lend commands as unknown; FAKE_BUSY=<n>: the
# first n RETURNs answer ERR BUSY; FAKE_SLOW=<n>: the first n LENDs are answered 1.2 s late (the command loop waits,
# as the engine's does while it reads a long prompt: the commands behind them wait too); FAKE_SLOW_RETURN=<n>: the
# same for RETURNs; FAKE_ODD=lend|return: the first such command gets an answer the server does not know; FAKE_DIE=1:
# a LEND ends the process without an answer; FAKE_LOG: every line it reads, one a line.
STAND_IN = r'''
import os, sys, threading, time

flag = "--vram-lend" in sys.argv
lend_on = flag and os.environ.get("FAKE_LEND", "1") == "1"
busy = [int(os.environ.get("FAKE_BUSY", "0") or 0)]
slow = [int(os.environ.get("FAKE_SLOW", "0") or 0)]
slow_ret = [int(os.environ.get("FAKE_SLOW_RETURN", "0") or 0)]
odd = [os.environ.get("FAKE_ODD", "")]
die = os.environ.get("FAKE_DIE") == "1"
log = os.environ.get("FAKE_LOG")
pool = {0: 4096, 1: 2048}                            # dev -> what it can lend with nothing out (MiB)
block = 64
leases, next_id = {}, [7]
lock = threading.Lock()
stop = threading.Event()


def out(s):
    with lock:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def lent(dev):
    return sum(m for d, m in leases.values() if d == dev)


def gen(f):
    ids = [int(x) for x in f[-1].split(",")][: int(f[1])]
    for i in ids:
        if stop.is_set():
            break
        time.sleep(0.03)
        out(f"T {i + 1000}")
    out(f"DONE {len(ids)} {len(ids)} 1.0 1.0 length")


def lend_cmd(f):
    if f[0] == "LENDABLE":
        if len(f) != 1:
            return ["ERR MALFORMED LENDABLE"]
        return [f"LENDABLE dev={d} lendable={pool[d] - lent(d)} lent={lent(d)}" for d in sorted(pool)] + ["END"]
    if f[0] == "LEND":
        if odd[0] == "lend":
            odd[0] = ""
            return ["ERR MAX plenty"]
        if len(f) != 3 or not f[1].isdigit() or not f[2].isdigit() or int(f[2]) == 0 or len(f[2]) > 9:
            return ["ERR MALFORMED LEND <dev> <mib>"]
        dev, mib = int(f[1]), -(-int(f[2]) // block) * block
        if dev not in pool:
            return [f"ERR NODEV {dev}"]
        if mib > pool[dev] - lent(dev):
            return [f"ERR MAX {pool[dev] - lent(dev)}"]
        i = next_id[0]
        next_id[0] += 5
        leases[i] = (dev, mib)
        return [f"LENT {i} {dev} {mib}"]
    if len(f) != 2:
        return ["ERR MALFORMED RETURN <id>"]
    if odd[0] == "return":
        odd[0] = ""
        return ["LENT 1 0 64"]
    i = int(f[1]) if f[1].isdigit() else -1
    if i not in leases:
        return [f"ERR UNKNOWN {f[1]}"]
    if busy[0] > 0:
        busy[0] -= 1
        return [f"ERR BUSY {i}"]
    del leases[i]
    return [f"RETURNED {i}"]


if flag:                                             # (the engine prints them only with --vram-lend)
    out("GPU-PCI dev=0 pci=0000:01:00.0")
    out("GPU-PCI dev=1 pci=0000:0A:00.0")
    out("GPU-PCI dev=2 pci=0000:0b:00.0")
out("READY 4096 stop")
for line in sys.stdin:
    if log:
        with open(log, "a") as fh:
            fh.write(line)
    f = line.split()
    if not f:
        continue
    if f[0] == "QUIT":
        break
    if f[0] == "STOP":
        stop.set()
        continue
    if f[0] == "GEN":
        stop.clear()
        threading.Thread(target=gen, args=(f,), daemon=True).start()
        continue
    if f[0] in ("LEND", "RETURN", "LENDABLE") and flag:
        if not lend_on:
            out(f"ERR unsupported command '{f[0]}' (GEN, GENI, STOP and QUIT)")
            continue
        if f[0] == "LEND" and die:
            os._exit(3)
        if f[0] == "LEND" and slow[0] > 0:
            slow[0] -= 1
            time.sleep(1.2)
        if f[0] == "RETURN" and slow_ret[0] > 0:
            slow_ret[0] -= 1
            time.sleep(1.2)
        for a in lend_cmd(f):
            out(a)
        continue
    out("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>")
'''


def stand_in(d: Path) -> str:
    exe = d / "engine"
    exe.write_text(f"#!{sys.executable}\n" + STAND_IN)
    exe.chmod(0o755)
    return str(exe)


class LendBase(unittest.TestCase):
    env: dict = {}
    args = ["--vram-lend"]

    def setUp(self):
        keep = mock.patch.dict(os.environ, {"STRATA_KEEP_LEFTOVERS": "1"})
        keep.start()
        self.addCleanup(keep.stop)
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        self.log = d / "lines.log"
        env = dict(os.environ, FAKE_LOG=str(self.log), **self.env)
        self.engine = StrataEngine(stand_in(d), list(self.args), env=env)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.engine.alive():
            self.engine.close()
        self.dir.cleanup()

    def call(self, method, path, body=None, raw=None, headers=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"null")

    def lend(self, gpu, mib):
        return self.call("POST", "/vram/lend", {"gpu": gpu, "mib": mib})

    def give_back(self, lease_id):
        return self.call("POST", "/vram/return", {"id": lease_id})

    def state(self):
        code, out = self.call("GET", "/vram/state")
        self.assertEqual(code, 200, out)
        return out

    def gpu(self, state, pci):
        return next(c for c in state["gpus"] if c["gpu"] == pci)

    def engine_lines(self, word):
        try:
            return [x.strip() for x in self.log.read_text().splitlines() if x.startswith(word + " ") or x == word + "\n"
                    or x == word]
        except OSError:
            return []

    def lend_lines(self):
        return self.engine_lines("LEND") + self.engine_lines("RETURN") + self.engine_lines("LENDABLE")


class Routes(LendBase):
    def test_lend_answers_the_lease(self):
        code, out = self.lend("0000:0a:00.0", 100)
        self.assertEqual(code, 200, out)
        self.assertEqual(set(out), {"id", "gpu", "mib"})
        self.assertIsInstance(out["id"], str)
        self.assertEqual((out["gpu"], out["mib"]), ("0000:0a:00.0", 128))   # (the engine's blocks: rounded up)
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 100"])           # the GPU's number from its GPU-PCI line

    def test_the_three_address_forms_in_any_case(self):
        # CUDA's form, NVML's 8-digit domain and lspci's form without a domain; answered in one spelling
        for gpu in ("0000:0A:00.0", "0000:0a:00.0", "00000000:0A:00.0", "00000000:0a:00.0", "0a:00.0", "0A:00.0"):
            code, out = self.lend(gpu, 64)
            self.assertEqual(code, 200, (gpu, out))
            self.assertEqual(out["gpu"], "0000:0a:00.0")
        self.assertEqual(self.engine_lines("LEND"), ["LEND 1 64"] * 6)

    def test_unknown_or_malformed_gpu_is_400(self):
        for gpu in ("0000:09:00.0", "0000:01:00.1", "gpu0", "", 3, None, ["0000:01:00.0"]):
            code, out = self.lend(gpu, 64)
            self.assertEqual(code, 400, (gpu, out))
            self.assertIn("error", out)
        code, out = self.call("POST", "/vram/lend", {"mib": 64})
        self.assertEqual(code, 400, out)
        self.assertIn('"gpu"', out["error"])
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_the_field_is_gpu_not_card(self):
        code, out = self.call("POST", "/vram/lend", {"card": "0000:01:00.0", "mib": 64})
        self.assertEqual(code, 400, out)
        self.assertIn('"gpu"', out["error"])
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_bad_mib_is_400(self):
        for mib in (0, -5, 1.5, 64.0, "100", True, False, None, [64]):
            code, out = self.lend("0000:01:00.0", mib)
            self.assertEqual(code, 400, (mib, out))
            self.assertIn("error", out)
        code, out = self.call("POST", "/vram/lend", {"gpu": "0000:01:00.0"})
        self.assertEqual(code, 400, out)
        self.assertEqual(self.engine_lines("LEND"), [])

    def test_a_mib_the_engine_refuses_as_malformed_is_400(self):
        code, out = self.lend("0000:01:00.0", 10 ** 12)
        self.assertEqual(code, 400, out)
        self.assertIn("error", out)

    def test_body_that_is_not_an_object_is_400(self):
        for raw in (b"not json", b"[1, 2]", b"7", b"\xff\xfe"):
            code, out = self.call("POST", "/vram/lend", raw=raw)
            self.assertEqual(code, 400, (raw, out))
            self.assertIsInstance(out["error"], str)
            code, out = self.call("POST", "/vram/return", raw=raw)
            self.assertEqual(code, 400, (raw, out))
        self.assertEqual(self.lend_lines(), [])

    def test_over_lendable_is_409_with_the_max(self):
        lendable = self.gpu(self.state(), "0000:01:00.0")["lendable"]
        self.assertEqual(lendable, 4096)
        code, out = self.lend("0000:01:00.0", lendable + 1)
        self.assertEqual((code, out), (409, {"max_mib": lendable}))
        self.assertEqual(self.lend("0000:01:00.0", 1024)[0], 200)
        code, out = self.lend("0000:01:00.0", 4096)
        self.assertEqual((code, out), (409, {"max_mib": self.gpu(self.state(), "0000:01:00.0")["lendable"]}))
        self.assertEqual(out["max_mib"], 3072)

    def test_a_gpu_that_does_not_lend_can_give_0(self):
        code, out = self.lend("0000:0B:00.0", 64)
        self.assertEqual((code, out), (409, {"max_mib": 0}))

    def test_state_shape(self):
        st = self.state()
        self.assertEqual(st, {"gpus": [{"gpu": "0000:01:00.0", "lendable": 4096, "lent": 0},
                                       {"gpu": "0000:0a:00.0", "lendable": 2048, "lent": 0},
                                       {"gpu": "0000:0b:00.0", "lendable": 0, "lent": 0}],
                              "leases": []})
        code, a = self.lend("0000:01:00.0", 512)
        self.assertEqual(code, 200)
        st = self.state()
        self.assertEqual(st["leases"], [{"id": a["id"], "gpu": "0000:01:00.0", "mib": 512}])
        self.assertEqual(self.gpu(st, "0000:01:00.0"), {"gpu": "0000:01:00.0", "lendable": 3584, "lent": 512})

    def test_leases_in_lend_order_agree_with_lent(self):
        got = [self.lend(g, m)[1] for g, m in (("0000:0a:00.0", 64), ("0000:01:00.0", 256), ("0000:0a:00.0", 128))]
        st = self.state()
        self.assertEqual([x["id"] for x in st["leases"]], [x["id"] for x in got])
        for g in st["gpus"]:
            self.assertEqual(sum(x["mib"] for x in st["leases"] if x["gpu"] == g["gpu"]), g["lent"], g)

    def test_two_leases_on_one_gpu_return_independently(self):
        code1, a = self.lend("0000:01:00.0", 256)
        code2, b = self.lend("0000:01:00.0", 512)
        self.assertEqual((code1, code2), (200, 200))
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(self.gpu(self.state(), "0000:01:00.0")["lent"], 768)
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 200, out)
        self.assertEqual(out, {"id": a["id"], "gpu": "0000:01:00.0", "mib": 256})
        st = self.state()
        self.assertEqual(st["leases"], [{"id": b["id"], "gpu": "0000:01:00.0", "mib": 512}])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], b["mib"])
        self.assertEqual(self.give_back(b["id"])[0], 200)
        st = self.state()
        self.assertEqual((st["leases"], self.gpu(st, "0000:01:00.0")["lent"]), ([], 0))

    def test_a_numeric_id_is_taken_too(self):
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 200)
        self.assertEqual(self.give_back(int(a["id"]))[0], 200)

    def test_unknown_id_is_404_and_not_sent(self):
        self.assertEqual(self.give_back("999")[0], 404)
        self.assertEqual(self.give_back("x9")[0], 404)
        self.assertEqual(self.engine_lines("RETURN"), [])
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(self.give_back(a["id"])[0], 200)
        code, out = self.give_back(a["id"])                      # (returned already)
        self.assertEqual(code, 404, out)
        self.assertIn("error", out)
        self.assertEqual(len(self.engine_lines("RETURN")), 1)

    def test_an_id_the_server_did_not_hand_out_is_404_and_not_sent(self):
        # a lease the engine has and the server never recorded: the server does not return it
        self.assertEqual(self.engine.lend_command("LEND 0 64"), ["LENT 7 0 64"])
        code, out = self.give_back("7")
        self.assertEqual(code, 404, out)
        self.assertEqual(self.give_back("999")[0], 404)
        self.assertEqual(self.engine_lines("RETURN"), [])
        self.assertEqual(self.gpu(self.state(), "0000:01:00.0")["lent"], 64)

    def test_an_id_from_before_a_restart_is_404(self):
        # the new process hands out the engine's ids from the start again: an old id never names its lease
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 200, a)
        self.engine.restart()
        code, b = self.lend("0000:01:00.0", 128)
        self.assertEqual(code, 200, b)
        self.assertNotEqual(b["id"], a["id"])
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 404, out)
        self.assertEqual(self.engine_lines("RETURN"), [])
        st = self.state()
        self.assertEqual(st["leases"], [{"id": b["id"], "gpu": "0000:01:00.0", "mib": 128}])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], 128)
        self.assertEqual(self.give_back(b["id"])[0], 200)
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])   # (the new process's own id for it)
        self.assertEqual(self.state()["leases"], [])

    def test_bad_id_is_400(self):
        for i in (None, True, 1.5, "", "1 2", "1\nLEND 0 64", ["1"]):
            code, out = self.give_back(i)
            self.assertEqual(code, 400, (i, out))
        self.assertEqual(self.call("POST", "/vram/return", {})[0], 400)
        self.assertEqual(self.lend_lines(), [])

    def test_restart_ends_every_lease(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 200)
        self.assertEqual(self.lend("0000:0a:00.0", 64)[0], 200)
        self.engine.restart()
        st = self.state()
        self.assertEqual(st["leases"], [])
        self.assertEqual([c["lent"] for c in st["gpus"]], [0, 0, 0])

    def test_engine_not_running(self):
        code, a = self.lend("0000:01:00.0", 64)
        self.engine.proc.kill()                       # (it died: an idle unload or a crash ends it the same way)
        self.engine.proc.wait()
        st = self.state()
        self.assertEqual(st["leases"], [])
        self.assertEqual([(c["lendable"], c["lent"]) for c in st["gpus"]], [(0, 0)] * 3)
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 503, out)
        self.assertEqual(self.give_back(a["id"])[0], 404)         # (it ended with the engine)

    def test_unloaded_engine(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 200)
        self.engine.unload()
        st = self.state()
        self.assertEqual(st, {"gpus": [{"gpu": "0000:01:00.0", "lendable": 0, "lent": 0},
                                       {"gpu": "0000:0a:00.0", "lendable": 0, "lent": 0},
                                       {"gpu": "0000:0b:00.0", "lendable": 0, "lent": 0}], "leases": []})
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 503)

    def test_lend_answers_never_reach_the_request_lines(self):
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 200)
        self.assertEqual(self.lend("0000:01:00.0", 99999)[0], 409)
        self.assertEqual(self.lend("0000:0b:00.0", 64)[0], 409)
        self.state()
        self.assertEqual(self.give_back(a["id"])[0], 200)
        time.sleep(0.1)
        left = []
        while not self.engine.lines.empty():
            left.append(self.engine.lines.get_nowait())
        self.assertEqual(left, [])

    def test_a_foreign_page_cannot_lend(self):
        code, out = self.call("POST", "/vram/lend", {"gpu": "0000:01:00.0", "mib": 64},
                              headers={"Origin": "http://elsewhere.example"})
        self.assertEqual(code, 403, out)
        self.assertEqual(self.lend_lines(), [])

    def test_no_other_routes(self):
        self.assertEqual(self.call("GET", "/vram/lend")[0], 404)
        self.assertEqual(self.call("GET", "/vram/return")[0], 404)
        self.assertEqual(self.call("POST", "/vram/state", {})[0], 404)


class Busy(LendBase):
    env = {"FAKE_BUSY": "1"}

    def test_busy_is_409_and_the_lease_stays(self):
        code, a = self.lend("0000:01:00.0", 128)
        self.assertEqual(code, 200)
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 409, out)
        self.assertIs(out.get("retry"), True)
        self.assertIn("error", out)
        st = self.state()
        self.assertEqual(st["leases"], [{"id": a["id"], "gpu": "0000:01:00.0", "mib": 128}])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], 128)
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 200, out)
        self.assertEqual(self.state()["leases"], [])


class OddAnswerToLend(LendBase):
    env = {"FAKE_ODD": "lend"}

    def test_an_answer_the_server_does_not_know_is_502(self):
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 502, out)
        self.assertIn("error", out)
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 200)   # (the next answer is the next command's)


class OddAnswerToReturn(LendBase):
    env = {"FAKE_ODD": "return"}

    def test_an_answer_the_server_does_not_know_is_502(self):
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 200)
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 502, out)
        self.assertIn("error", out)
        self.assertEqual(self.state()["leases"], [a])             # (nothing came back: still lent)
        self.assertEqual(self.give_back(a["id"])[0], 200)


class EngineEndsDuringALend(LendBase):
    env = {"FAKE_DIE": "1"}

    def test_503(self):
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 503, out)
        self.assertIn("error", out)
        st = self.state()
        self.assertEqual((st["leases"], [c["lent"] for c in st["gpus"]]), ([], [0, 0, 0]))


class LateAnswers(LendBase):
    """A LEND the engine answers after the server stopped waiting (the engine takes the lend commands between its
    steps, and one step can read a long prompt): the answer is that LEND's, never the next command's, and its lease
    goes back to the engine."""
    env = {"FAKE_SLOW": "1"}

    def setUp(self):
        super().setUp()
        self.svc.vram_lend.answer_s = 0.5

    def given_back(self, returns=1):
        """/vram/state once the late lease is back (the engine read RETURN and nothing is listed any more)."""
        end = time.monotonic() + 10
        while time.monotonic() < end:
            if len(self.engine_lines("RETURN")) >= returns:
                st = self.state()
                if not st["leases"]:
                    return st
            time.sleep(0.05)
        self.fail(f"the late lease did not go back: {self.engine_lines('RETURN')}, {self.state()}")

    def test_late_lend_is_given_back(self):
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 503, out)
        self.assertIn("give it back", out["error"])
        st = self.given_back()
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])
        self.assertEqual([c["lent"] for c in st["gpus"]], [0, 0, 0])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lendable"], 4096)

    def test_retry_after_a_late_lend_holds_one_lease(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 503)
        self.svc.vram_lend.answer_s = 10
        code, b = self.lend("0000:01:00.0", 128)
        self.assertEqual(code, 200, b)
        self.assertEqual((b["gpu"], b["mib"]), ("0000:01:00.0", 128))
        self.assertNotEqual(b["id"], "7")
        end = time.monotonic() + 10
        while time.monotonic() < end:
            st = self.state()
            if self.engine_lines("RETURN") and len(st["leases"]) == 1:
                break
            time.sleep(0.05)
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])       # the late one, not the retry's
        self.assertEqual(st["leases"], [b])
        self.assertEqual(self.gpu(st, "0000:01:00.0")["lent"], 128)

    def test_state_after_a_late_lend_reads_the_engine(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 503)
        self.svc.vram_lend.answer_s = 10
        st = self.state()
        self.assertEqual(self.gpu(st, "0000:0a:00.0"), {"gpu": "0000:0a:00.0", "lendable": 2048, "lent": 0})
        first = self.gpu(st, "0000:01:00.0")
        self.assertEqual(first["lendable"] + first["lent"], 4096)
        self.assertEqual(sum(x["mib"] for x in st["leases"]), first["lent"])   # listed while the engine lends it
        st = self.given_back()
        self.assertEqual(self.gpu(st, "0000:01:00.0"), {"gpu": "0000:01:00.0", "lendable": 4096, "lent": 0})


class LateLeaseStillInUse(LateAnswers):
    """The server's own RETURN of a late lease answers ERR BUSY: it is sent again at the next API call."""
    env = {"FAKE_SLOW": "1", "FAKE_BUSY": "1"}

    def test_late_lend_is_given_back(self):
        self.assertEqual(self.lend("0000:01:00.0", 64)[0], 503)
        end = time.monotonic() + 10
        while time.monotonic() < end and not self.engine_lines("RETURN"):
            time.sleep(0.05)
        time.sleep(0.2)                                   # (its ERR BUSY is in)
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7"])
        self.state()                                      # this call sends it again ...
        self.given_back(returns=2)                        # ... and the engine has it back
        self.assertEqual(self.engine_lines("RETURN"), ["RETURN 7", "RETURN 7"])

    test_retry_after_a_late_lend_holds_one_lease = None
    test_state_after_a_late_lend_reads_the_engine = None


class LateReturn(LendBase):
    env = {"FAKE_SLOW_RETURN": "1"}

    def test_a_return_not_answered_in_time_is_503_and_its_answer_still_counts(self):
        code, a = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 200)
        self.svc.vram_lend.answer_s = 0.5
        code, out = self.give_back(a["id"])
        self.assertEqual(code, 503, out)
        self.assertIn("error", out)
        self.svc.vram_lend.answer_s = 10
        end = time.monotonic() + 10
        while time.monotonic() < end and self.state()["leases"]:
            time.sleep(0.05)
        st = self.state()
        self.assertEqual((st["leases"], self.gpu(st, "0000:01:00.0")["lent"]), ([], 0))
        self.assertEqual(self.give_back(a["id"])[0], 404)


class WhileRequestsRun(LendBase):
    """A lend, a return and the state while a request runs: not behind the request queue, and the answers stay out of
    the request's own lines."""

    def test_one_request(self):
        self.svc.fifo.acquire()                       # (the request queue is held by "requests")
        try:
            ids = list(range(1, 61))
            got = {}

            def one():
                got["toks"] = [t for t in self.engine.generate(ids, len(ids), {}, threading.Event()) if t is not None]
            th = threading.Thread(target=one)
            th.start()
            time.sleep(0.2)
            t0 = time.monotonic()
            code, a = self.lend("0000:01:00.0", 64)
            self.assertEqual(code, 200, a)
            code, out = self.lend("0000:01:00.0", 99999)
            self.assertEqual(code, 409, out)
            st = self.state()
            self.assertEqual(st["leases"], [{"id": a["id"], "gpu": "0000:01:00.0", "mib": 64}])
            self.assertEqual(self.give_back(a["id"])[0], 200)
            self.assertEqual(self.give_back(a["id"])[0], 404)
            self.assertLess(time.monotonic() - t0, 1.5)
            self.assertTrue(th.is_alive())            # (all of it while the request ran)
            th.join(30)
            self.assertEqual(got.get("toks"), [i + 1000 for i in ids])
        finally:
            self.svc.fifo.release()


class Unsupported(LendBase):
    """An engine started with the option that refuses the lend commands as unknown."""
    env = {"FAKE_LEND": "0"}

    def test_engine_without_lending(self):
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 501, out)
        self.assertIn("error", out)
        self.assertEqual(self.give_back("7")[0], 501)
        self.assertEqual(self.state(), {"gpus": [{"gpu": "0000:01:00.0", "lendable": 0, "lent": 0},
                                                 {"gpu": "0000:0a:00.0", "lendable": 0, "lent": 0},
                                                 {"gpu": "0000:0b:00.0", "lendable": 0, "lent": 0}],
                                        "leases": []})
        # a request's own lines are still the request's
        ids = [1, 2, 3]
        self.assertEqual([t for t in self.engine.generate(ids, 3, {}, threading.Event()) if t is not None],
                         [1001, 1002, 1003])


# nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader as the rig answers it (numbered by PCI bus)
SMI = ("0, 00000000:01:00.0\n1, 00000000:02:00.0\n2, 00000000:05:00.0\n3, 00000000:09:00.0\n"
       "4, 00000000:0A:00.0\n")


def fake_smi(test, text=SMI, missing=False) -> list:
    """nvidia-smi answers `text` (missing: it is not installed) for the test's life, read afresh (nothing cached
    from an earlier test); -> the list its calls go to."""
    real, calls = subprocess.run, []

    def run(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "nvidia-smi":
            calls.append(list(cmd))
            if missing:
                raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")
            return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")
        return real(cmd, *a, **k)
    for p in (mock.patch("subprocess.run", run), mock.patch.object(gpu_map, "_smi", None, create=True)):
        p.start()
        test.addCleanup(p.stop)
    return calls


# the run's cards in the engine's order (its GPU 0 first): the 4060 Ti on 0a:00.0, then the 3090 on 01:00.0
RUN = [{"gpu": "0000:0a:00.0", "lendable": 0, "lent": 0}, {"gpu": "0000:01:00.0", "lendable": 0, "lent": 0}]


class OptionOff(LendBase):
    """Without --vram-lend (the config's "vram_lend" not true): every lend and return answers 501, /vram/state lists
    the run's GPUs with 0 and 0, and nothing goes to the engine.  The engine prints no GPU-PCI lines then: the server
    reads the cards' addresses from nvidia-smi, in the engine's card order."""
    args: list = []

    def setUp(self):
        super().setUp()
        self.smi = fake_smi(self)
        self.svc.engine_gpus = [4, 0]

    def test_501_and_0_0(self):
        self.assertFalse(self.engine.lending)
        code, out = self.lend("0000:01:00.0", 64)
        self.assertEqual(code, 501, out)
        self.assertIsInstance(out["error"], str)
        self.assertEqual(self.give_back("7")[0], 501)
        self.assertEqual(self.give_back(7)[0], 501)
        self.assertEqual(self.engine.lend_gpus, [])
        self.assertEqual(self.state(), {"gpus": RUN, "leases": []})
        self.assertEqual(self.state(), {"gpus": RUN, "leases": []})
        self.assertEqual(self.smi, [["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"]])  # once
        self.assertEqual(self.lend_lines(), [])
        ids = [1, 2, 3]
        self.assertEqual([t for t in self.engine.generate(ids, 3, {}, threading.Event()) if t is not None],
                         [1001, 1002, 1003])

    def test_lend_shaped_lines_stay_the_requests(self):
        # with the option off the line reader is as before: nothing is taken off the request's lines
        self.engine.proc.stdin.write("NOPE\n")
        self.engine.proc.stdin.flush()
        line = self.engine.lines.get(timeout=10)
        self.assertTrue(line.startswith("ERR expected"), line)

    def test_engine_not_running(self):
        # an idle unload (or before the first load): the run's GPUs, each 0 and 0
        self.engine.close()
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.state(), {"gpus": RUN, "leases": []})

    def test_a_card_nvidia_smi_does_not_list_is_left_out(self):
        self.svc.engine_gpus = [4, 7, 0]
        self.assertEqual(self.state(), {"gpus": RUN, "leases": []})

    def test_odd_nvidia_smi_lines_are_skipped(self):
        self.smi = fake_smi(self, "0, 00000000:01:00.0\nNo devices were found\n4, -\n, 0a:00.0\n\n4, 0A:00.0\n")
        self.assertEqual(self.state(), {"gpus": RUN, "leases": []})


class OptionOffNoAddresses(LendBase):
    """Without --vram-lend and with no way to read the cards' addresses: no GPUs, as for an engine that reports none,
    and no error."""
    args: list = []

    def test_no_nvidia_smi(self):
        smi = fake_smi(self, missing=True)
        self.svc.engine_gpus = [4, 0]
        self.assertEqual(self.state(), {"gpus": [], "leases": []})
        self.assertEqual(self.state(), {"gpus": [], "leases": []})
        self.assertEqual(len(smi), 1)                 # (not asked again)

    def test_nvidia_smi_fails(self):
        def run(cmd, *a, **k):
            raise subprocess.TimeoutExpired(cmd, 10)
        self.svc.engine_gpus = [4, 0]
        with mock.patch.object(gpu_map, "_smi", None, create=True), mock.patch("subprocess.run", run):
            self.assertEqual(self.state(), {"gpus": [], "leases": []})
        smi = fake_smi(self)                          # (a timeout is not kept: the next state asks again)
        self.assertEqual(self.state(), {"gpus": RUN, "leases": []})
        self.assertEqual(len(smi), 1)

    def test_amd_backend_and_cpu_run(self):
        smi = fake_smi(self)
        self.svc.backend, self.svc.engine_gpus = "hip", [4, 0]      # (HIP numbers its cards itself)
        self.assertEqual(self.state(), {"gpus": [], "leases": []})
        self.svc.backend, self.svc.engine_gpus = None, []           # a CPU run: no cards
        self.assertEqual(self.state(), {"gpus": [], "leases": []})
        self.assertEqual(smi, [])


class OptionOnLinesWin(LendBase):
    """With --vram-lend the engine's GPU-PCI lines name the GPUs, also once the engine has stopped: nvidia-smi is not
    asked."""

    def test_lines_win(self):
        smi = fake_smi(self)
        self.svc.engine_gpus = [4, 0]
        want = ["0000:01:00.0", "0000:0a:00.0", "0000:0b:00.0"]
        self.assertEqual([g["gpu"] for g in self.state()["gpus"]], want)
        self.engine.close()
        self.assertEqual(self.state(), {"gpus": [{"gpu": a, "lendable": 0, "lent": 0} for a in want], "leases": []})
        self.assertEqual(smi, [])


class NoLendingEngine(unittest.TestCase):
    def test_mock_engine(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def call(method, path, body=None):
                req = urllib.request.Request(base + path, method=method,
                                             data=json.dumps(body).encode() if body is not None else None)
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, json.loads(e.read())
            self.assertEqual(call("GET", "/vram/state"), (200, {"gpus": [], "leases": []}))
            self.assertEqual(call("POST", "/vram/lend", {"gpu": "0000:01:00.0", "mib": 64})[0], 501)
            self.assertEqual(call("POST", "/vram/return", {"id": "7"})[0], 501)
        finally:
            httpd.shutdown()
            httpd.server_close()


class InOrder(unittest.TestCase):
    """The answer matching on its own: an answer is the oldest command's, also when its caller stopped waiting."""

    def test_a_caller_that_stopped_waiting_keeps_its_answer(self):
        sent = []
        calls = vram_lend.Calls(sent.append)
        seen = []
        a = calls.send("LEND 0 64", lambda got, waited: seen.append(("a", list(got), waited)))
        self.assertFalse(calls.wait(a, 0.05))             # its caller gives up
        b = calls.send("LENDABLE", lambda got, waited: seen.append(("b", list(got), waited)))
        calls.answer("LENT 7 0 64\n")                     # the first answer: a's, not b's
        calls.answer("LENDABLE dev=0 lendable=4032 lent=64\n")
        self.assertFalse(b.done.is_set())                 # (LENDABLE ends at END)
        calls.answer("END\n")
        self.assertTrue(calls.wait(b, 1))
        self.assertEqual(sent, ["LEND 0 64", "LENDABLE"])
        self.assertEqual(seen, [("a", ["LENT 7 0 64"], False),
                                ("b", ["LENDABLE dev=0 lendable=4032 lent=64", "END"], True)])
        self.assertEqual(b.got, ["LENDABLE dev=0 lendable=4032 lent=64", "END"])

    def test_the_engine_ending_ends_every_waiting_call(self):
        calls = vram_lend.Calls(lambda line: None)
        a = calls.send("LEND 0 64")
        calls.close()
        self.assertTrue(calls.wait(a, 1))
        self.assertIsNone(a.got)
        self.assertIsNone(calls.send("LENDABLE"))

    def test_a_write_that_fails_is_no_call(self):
        def broken(line):
            raise OSError("closed")
        calls = vram_lend.Calls(broken)
        self.assertIsNone(calls.send("LENDABLE"))
        calls.answer("END\n")                             # (nothing waits: dropped)


class Answers(unittest.TestCase):
    """Which engine lines are lend answers (taken off the request's lines) and which are not."""

    def test_is_answer(self):
        for line in ("LENT 7 0 128\n", "RETURNED 7\n", "LENDABLE dev=0 lendable=4096 lent=0\n", "END\n", "END\r\n",
                     "ERR MAX 4096\n", "ERR NODEV 3\n", "ERR BUSY 7\n", "ERR UNKNOWN x9\n",
                     "ERR MALFORMED LEND <dev> <mib>\n", "ERR MALFORMED RETURN <id>\n", "ERR MALFORMED LENDABLE\n",
                     "ERR unsupported command 'LEND' (GEN, GENI, STOP and QUIT)\n",
                     "ERR unsupported command 'LENDABLE' (GEN, GENI, LOGP, STOP and QUIT)\n"):
            self.assertTrue(vram_lend.is_answer(line), line)
        for line in ("T 5\n", "DONE 3 3 1.0 1.0 length\n", "ERR non-finite\n", "ERR 3 non-finite\n", "RELENT 123\n",
                     "ENDED\n", "PP 512 4096 120.0\n", "BT 0 5\n", "BDONE 0 3 3 1.0 1.0 stop\n",
                     "VRAM reserve_mib=700 gpu=0 expert_slots=10\n", "VRAM-DONE\n", "GPU-PCI dev=0 pci=0000:01:00.0\n",
                     "ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>\n",
                     "ERR verify: timed out at layer 3\n", "ERR an adaptive refill failed\n",
                     "ERR VRAM: not while batch slots are decoding\n",
                     "ERR prompt (10 tokens) + max_new (5) exceeds the context (8)\n"):
            self.assertFalse(vram_lend.is_answer(line), line)


class Addresses(unittest.TestCase):
    def test_pci_addr_forms(self):
        for s in ("0000:0A:00.0", "0000:0a:00.0", "00000000:0A:00.0", "0a:00.0", "0A:00.0", " 0a:00.0\n", "0a:00"):
            self.assertEqual(gpu_map.pci_addr(s), "0000:0a:00.0", s)
        self.assertEqual(gpu_map.pci_addr("0001:41:1f.7"), "0001:41:1f.7")
        for s in ("", "gpu0", "0a", "0000:0a:20.0", "0000:0a:00.8", "0000:100:00.0", "10000:0a:00.0", "0000:0g:00.0",
                  "0000::00.0", "0:0:0:0", None, 3, ["0a:00.0"]):
            self.assertIsNone(gpu_map.pci_addr(s), s)

    def test_parse_the_per_gpu_line(self):
        self.assertEqual(gpu_map.parse("GPU-PCI dev=1 pci=0000:0A:00.0\n"),
                         {"dev": 1, "pci": "0000:0a:00", "addr": "0000:0a:00.0"})
        self.assertEqual(gpu_map.parse("GPU-PCI dev=0 pci=00000000:01:00.0 name=x"),
                         {"dev": 0, "pci": "0000:01:00", "addr": "0000:01:00.0"})
        for line in ("GPU-PCI pci=0000:01:00.0", "GPU floor pre+post+head 1.00 ms per token",
                     "VRAM reserve_mib=700 gpu=0", "INFO dev=0 pci=0000:01:00.0", ""):
            self.assertIsNone(gpu_map.parse(line), line)

    def test_nvidia_smi_addresses(self):
        fake_smi(self)
        self.assertEqual(gpu_map.nvidia_smi(), {0: "0000:01:00.0", 1: "0000:02:00.0", 2: "0000:05:00.0",
                                                3: "0000:09:00.0", 4: "0000:0a:00.0"})


class ConfigKey(unittest.TestCase):
    def test_the_key_passes_the_engine_flag(self):
        cfg = {"args": ["--max-context", "8192"]}
        self.assertEqual(engine_args(cfg), ["--max-context", "8192"])           # off unless asked
        self.assertEqual(engine_args({**cfg, "vram_lend": True}), ["--max-context", "8192", "--vram-lend"])
        self.assertEqual(engine_args({**cfg, "vram_lend": "yes"}), ["--max-context", "8192"])
        self.assertEqual(engine_args({"args": ["--vram-lend"], "vram_lend": True}), ["--vram-lend"])

    def test_the_key_is_editable(self):
        cfg = {"args": ["--max-context", "8192"]}
        self.assertIs(runconfig.check("vram_lend", True, cfg), True)
        with self.assertRaises(ValueError):
            runconfig.check("vram_lend", "yes", cfg)
        new, changed = runconfig.apply(cfg, {"vram_lend": True})
        self.assertEqual((changed, engine_args(new)), (["vram_lend"], ["--max-context", "8192", "--vram-lend"]))


class SourceScan(unittest.TestCase):
    def test_vram_json_names_no_card(self):
        # the /vram JSON names a GPU "gpu" (and the list "gpus"): no "card" key in a dict literal or a lookup
        import re
        key = re.compile(r"""["']cards?["']\s*:|\[\s*["']cards?["']\s*\]|\.get\(\s*["']cards?["']""")
        text = (ROOT / "serve" / "vram_lend.py").read_text(encoding="utf-8")
        self.assertEqual(key.findall(text), [])


if __name__ == "__main__":
    unittest.main()

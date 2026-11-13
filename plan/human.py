"""Your own play as an agent (agent.py --human): one agent of the game's pool is driven by your keyboard or
gamepad instead of the planner, so the recording (--wm-collect) has exactly the agents' form: the same
character mode, the same 10 inputs (mpc.candidates.ACTIONS), the same frame-by-frame time-step lock, deaths and
finishes. The hidden real Meat Boy does not move (realgame.js); the camera follows your agent.

Input is read once per game frame, only while the game window is in front (anything typed elsewhere does
nothing). Both devices count at once:
- keyboard: left = Left arrow or A, right = Right arrow or D, jump = Space, Up arrow or W, run = Shift or Ctrl.
  Never S: realgame.js finishes the level the game's way on S.
- gamepad (XInput, the first pad): left / right = d-pad or the left stick (past STICK_DEAD), jump = A or B, run =
  X, Y, a bumper or a trigger (past TRIGGER_DEAD).
Mapped to the action table as the agents' inputs are: left and right together is neither; run without a
direction is no run (the table has no such action). Jump held against a wall is let go and pressed again by
realgame.js after WALL_REPRESS frames, as for every agent: holding jump there wall-jumps."""
from __future__ import annotations

import ctypes
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mpc.candidates import act                                  # noqa: E402

GAME_EXE = "SuperMeatBoy.exe"
VK_LEFT, VK_UP, VK_RIGHT, VK_SPACE = 0x25, 0x26, 0x27, 0x20
VK_SHIFT, VK_CONTROL = 0x10, 0x11
KEYS = {"left": (VK_LEFT, ord("A")), "right": (VK_RIGHT, ord("D")), "jump": (VK_SPACE, VK_UP, ord("W")),
        "run": (VK_SHIFT, VK_CONTROL)}
PAD_LEFT, PAD_RIGHT = 0x0004, 0x0008
PAD_JUMP = 0x1000 | 0x2000                       # A, B
PAD_RUN = 0x4000 | 0x8000 | 0x0100 | 0x0200      # X, Y, left / right bumper
STICK_DEAD = 12000                               # of 32767
TRIGGER_DEAD = 40                                # of 255


class _PadState(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("buttons", ctypes.c_uint16), ("lt", ctypes.c_uint8),
                ("rt", ctypes.c_uint8), ("lx", ctypes.c_int16), ("ly", ctypes.c_int16), ("rx", ctypes.c_int16),
                ("ry", ctypes.c_int16)]


def to_action(left: bool, right: bool, jump: bool, run: bool) -> int:
    """The action index of these buttons (module docstring's mapping)."""
    d = (-1 if left else 0) + (1 if right else 0)
    return act(d, jump=jump, run=run)


class HumanInput:
    """Reads the keyboard and the first XInput pad (Windows); action() is this frame's action index."""

    def __init__(self, exe: str = GAME_EXE):
        self.exe = exe.lower()
        self.user32 = ctypes.WinDLL("user32")
        self.kernel32 = ctypes.WinDLL("kernel32")
        self.user32.GetAsyncKeyState.restype = ctypes.c_int16
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p
        self.user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        self.kernel32.OpenProcess.restype = ctypes.c_void_p
        self.kernel32.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_wchar_p,
                                                             ctypes.POINTER(ctypes.c_uint32)]
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.xinput = None
        for name in ("XInput1_4", "xinput1_3", "XInput9_1_0"):
            try:
                self.xinput = ctypes.WinDLL(name)
                break
            except OSError:
                continue
        self.pad = _PadState()
        self._fg = (None, False)                  # (foreground window, it is the game)
        self.frames = self.pressed = 0            # frames read / frames with any input (the run's summary)

    def focused(self) -> bool:
        """The game's window is in front (cached per foreground window)."""
        hwnd = self.user32.GetForegroundWindow()
        if hwnd == self._fg[0]:
            return self._fg[1]
        pid = ctypes.c_uint32(0)
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        ok = False
        h = self.kernel32.OpenProcess(0x1000, False, pid.value)          # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            buf = ctypes.create_unicode_buffer(1024)
            n = ctypes.c_uint32(len(buf))
            if self.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
                ok = os.path.basename(buf.value).lower() == self.exe
            self.kernel32.CloseHandle(h)
        self._fg = (hwnd, ok)
        return ok

    def _key(self, which: str) -> bool:
        return any(self.user32.GetAsyncKeyState(vk) & 0x8000 for vk in KEYS[which])

    def buttons(self) -> tuple[bool, bool, bool, bool]:
        """(left, right, jump, run) from both devices; all False when the game is not in front."""
        if not self.focused():
            return False, False, False, False
        left, right, jump, run = self._key("left"), self._key("right"), self._key("jump"), self._key("run")
        if self.xinput is not None and self.xinput.XInputGetState(0, ctypes.byref(self.pad)) == 0:
            p = self.pad
            left |= bool(p.buttons & PAD_LEFT) or p.lx < -STICK_DEAD
            right |= bool(p.buttons & PAD_RIGHT) or p.lx > STICK_DEAD
            jump |= bool(p.buttons & PAD_JUMP)
            run |= bool(p.buttons & PAD_RUN) or p.lt > TRIGGER_DEAD or p.rt > TRIGGER_DEAD
        return left, right, jump, run

    def action(self) -> int:
        a = to_action(*self.buttons())
        self.frames += 1
        self.pressed += a != 0
        return a

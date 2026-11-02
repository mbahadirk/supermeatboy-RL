# Super Meat Boy RL: project history

What was built, which problems came up and how each was solved, from the first look at the game files to
PPO agents playing as real Meat Boys inside the game. Newest work is at the end. Addresses are for the
Epic Games Store build of `SuperMeatBoy.exe` (32-bit, image base 0x400000).

## Where the project stands

- `realgame.js` is a Frida script that runs inside the game. Up to 200 real Meat Boy characters play a level
  at the same time ("agents"), in waves. Each frame it sends their state and the level's live state to
  Python and waits for their inputs (lockstep).
- `realgame.py` is the trainer: PPO (PyTorch, CPU) with a 282-input observation and 10 actions, rewards
  from reward routes, waves, saving, the monitor.
- `realpath.py` plans the reward routes on the live tile map (A* with a jump model) and measures progress
  along them (closeness fields).
- `monitor.py` is a pygame window: map, routes, agents, hazards, the leader's action probabilities, curves.

```bash
python smbrl/realgame.py --agents 100 --end-after 5 --load smbrl/runs/<run>
```

Stop with Ctrl+C or by creating `runs/<name>/STOP`; never kill the process (that crashed the game once).

---

## 1. Reverse engineering the game (before any RL)

| Problem | What was found / done |
|---|---|
| How does Meat Boy move? | Physics is data-driven: `CharProps/*.props` (11 floats). Real values: gravity 550, ground accel 13/frame, air accel 8, jump 300, wall jump 350 up / 300 out, wall slide gravity x0.6. Calibrated on ~51k live frames; a clone matched the game's jump arc exactly. |
| Where are the levels? | `gamedata.dat` is an archive (TOC decoded); 591 `.smbl` levels extracted. `tools/smbl_extract.py` decodes tiles (RLE layers, tile word bits: id, special bit 8, rotation, flip, shape 18-19), objects, movement paths, chapter order. |
| Live state | Frida hooks on the character update (`FUN_00520d30`): position +0x84/+0x88, velocity +0x9c/+0xa0, state +0x7b0, flags +0x7b4 (bit 1 airborne, 0x200 on a wall). The tile map object gives the collision layer; the collision world `[0x7035c4]` holds the level pieces (+0x124). |
| Which pieces kill? | The world's type table (row flags: 0x06 goal, bit 0 lethal, 0x0c bandage, 0x44 key). Hazard sizes come from the game itself (`FUN_004ca860`), times the piece scale, turned by its angle (+0x2c). |

## 2. A Python clone, then into the real game

- First a clean-room clone of the physics with SB3 PPO (`engine.py`, `env.py`, `train.py`). It solved
  levels like 1-1, 2-4 and 3-1, and a trained model even won 1-1 in the real game through a virtual XInput
  pad.
- **Decision:** the clone was dropped. Agents must be real Meat Boys in the real game, not a copy of it.
- **How:** the game keeps 40 spare Meat Boys per level for its replay. `realgame.js` takes them over
  (mode +0x9e0 = 2, our own), feeds each one inputs through the character's input handlers, and steps
  them inside the hidden main character's update.

| Problem | Solution |
|---|---|
| Agents teleported onto the goal after a PPO update | The game's time step includes the time it waited for Python. Long work now happens only at level info / wave end, and agents are held still for one frame after any long wait (> 80 ms). |
| Agents drawn where they died | The render position (+0x824) is copied from the position on wave reset. |
| Reward routes not visible | Routes are drawn into the game with Direct3D 9 at Present, using the camera matrix captured mid-render (0x4ac6a9). |
| Killing the trainer crashed the game | Safe stop through the STOP file and a Ctrl+C handler that finishes cleanly. |
| A second agent (Gemini) had edited the project | Its stagnation penalty outweighed the win reward and its draw hook broke drawing; both fixed. |

## 3. Reward routes and what agents can see

| Problem (reported in training) | Solution |
|---|---|
| Routes went through walls (3-2) | Half blocks are walls; slopes are open only through their open corner (shape decode checked: 1329 vs 4 tiles). |
| Routes went through saws and spikes | Hazard modes, first that works: hazards closed, moving sweeps crossable with timing, everything crossable at a high cost, squeeze through half tiles. A cell is closed when the body (12x12) would touch a hazard box there, so tight gaps the body fits through stay open. |
| Routes "flew" where the player must wall-jump | Jump model: after leaving a floor or wall a route may rise at most 5 tiles and move 12 sideways, then only fall. A cell beside a wall counts as a surface, so climbing by wall jumps is allowed. Higher jumps, then no limits, only if nothing fits. |
| Keys and bandages | Routes visit the keys one by one (nearest reachable first), then the goal; a separate route goes through the bandage. Bandage pickup stays disabled for agents so the save file's records are not touched. |
| One agent's key opened locks for everyone | Per-agent keys: the game script sets the lock tiles to each agent's own key state right before that agent's update and restores them afterwards. |
| Broken walls still drawn intact | Every frame the game script sends the tile words that changed; the map and monitor are live. |
| Agents blind to moving and thrown saws | Hazards are re-read every frame. The two nearest moving hazards (offset and velocity) are inputs. Shots in flight are read from the launcher array (+0x2c30, slots of 0xf8 bytes). |
| Hazard angle read as scale | Piece scale is at +0x24/+0x28 and angle at +0x2c. |
| Single route only | Up to 3 routes plus a bandage route; agents follow the route they are furthest along; a route nobody gets far on keeps fewer agents. |
| Planning took up to 52 s | Precomputed moves, plain lists, an expansion cap and a time budget: worst level about 8 s, all 220 chapter levels get a route. |
| Fans, wind and orbs (2-4) | Fan and orb arrays read from the world; the wind force is computed like the game (`FUN_004c40e0`). Inputs: wind here, in a wind column, nearest orb. Wind columns count as ground for routes. |
| Accelerator surfaces (3-6) | Tile id 98 is marked in the map (what it does was not traced). |
| Portals | Paired in piece order (`FUN_004c8c10`); routes go through them. Agent teleports seen 3 times also become links. |
| Walls and floors looked the same to the network | A second 11x11 map marks what each solid cell offers: floor +1, wall -1, ledge corner +0.5, ceiling -0.5, inside +0.25. Plus wall-left / wall-right / wall-slide inputs. |
| False portal links | Some deaths make the game put the agent back on the spawn; those jumps are counted as deaths, never as portals. |

## 4. 2026-11-02: more agents, a better reward, platforms

### 100 agents

- **Problem:** `--agents 100` always gave 40.
- **Cause 1:** extra characters were only built at a level load, and training always started mid-level.
  Fixed by building them when attaching too.
- **Cause 2:** building them still failed with "system error". The game's clone constructor writes a log
  line with `OutputDebugStringA`, which raises an exception that Windows normally handles inside the
  call. Frida's default for native calls catches every exception, including that one.
- **Fix:** the character factory is called with `exceptions: 'propagate'`.

### Reward by closeness instead of "stay on the line"

- **Problem:** agents hugged the route line. Progress only counted from visible route points, there was a
  penalty beyond 2 tiles, and another route only counted within 2.5 tiles of it.
- **Fix:** each route has a closeness field. A spot is worth "progress of a nearby route point minus the
  walking distance to it". Walls block it and climbing a tile counts double.
- **Exploit found right after:** standing in a corner near a far stretch of the route earned up to 11 tiles
  of progress. The distance cost was only half a tile per tile, and a "3 tiles past your best" window crept
  forward every frame.
- **Fix:** a tile of distance now costs a full tile, every solid tile is a wall, and the window was
  replaced by route legs. The route counts only up to the next key the agent has not touched. A spot's
  value is fixed, so waiting earns nothing (measured: 0 on every level).

### Getting stuck, jumping in place, exploring

- **Stuck:** no new best progress and no new 2x2-tile square for 1.5 s costs a little every frame,
  rising, together with jump spam capped at 4 per stuck stretch. The cap is below a death, so dying is
  never the cheaper way out.
- **Exploring:** the first visit in a life to a 2x2-tile square pays a bonus, more where few agents have
  been.
- **Inputs:** two new inputs say how long the agent has been stuck and how often it jumped there.
- **Tried and dropped:** killing an agent that stood still for 1 s made levels that need waiting
  impossible.
- **Death penalty:** raised from 5 to 15.

### Agents turning back for no reason

- **Cause:** progress only paid new best ground, so going back was free. And stepping 2 tiles away reset
  the stuck timer, which made walking back the cheapest way to wait for a saw.
- **Fix:** progress pays both ways every frame, so going back costs exactly what going forward paid. The
  stuck timer resets only on new progress or a new square. Fitness still uses the best progress.

### Route splitting

- **Problem:** a route counted as "different" if it was more than 1 cell from earlier ones. So 66% of all
  extra routes (189 of 286 on the 220 chapter levels) were copies 2 cells beside the first route.
- **Fix:** a route cell runs the same way as an earlier route when it is within 4 tiles of it with no wall
  between. The search is also pushed that far.
- **Result:** 136 extra routes, none of them copies, every level still has a route.

### Moving platforms

- **Problem:** agents could not see them.
- **Found in the game:** platforms are in their own world array (count +0x2dc4, data +0x2dd4, 0x14
  bytes: piece, three tile shapes, upright flag). Collision (around 0x4cde59) tests a 60x20 bar (20x60
  upright).
- **Agents now see:** platform cells in the map, floor in the surface map, and the nearest platform's
  offset and speed.
- **Routes:** a platform seen in place for 4 s is a wall. Where moving platforms go is crossed only with
  timing, so the planner looks for another way first. Spots above platforms count as ground.

### Lava pits

- **Problem:** pits are made of a few stacked lava pieces whose boxes leave gaps, so the map showed open
  space inside the pit.
- **Fix:** open cells that cannot be reached from the spawn, goal, keys or portals without crossing a
  wall or a fixed hazard are filled in, as hazard or solid. Pockets over 150 cells stay open.

### Wall climbing

- **Problem:** a wall jump needs a new jump press, and agents could not see whether their jump key was
  held.
- **Fix:** jump kept held for 6+ frames in the air against a wall is released for one frame and pressed
  again. A new "jump held" input shows the key state.

---

## Rewards now (per agent, per life)

| Term | Amount |
|---|---|
| Progress (closeness to routes) | +1 per tile, both ways |
| Key | +10 once per key |
| Exploring | +0.25 per new 2x2-tile square, less where many agents went |
| Running toward the route on the ground | up to +0.006 per frame |
| Jump from the ground | -0.08 |
| Stuck (no progress, no new square for 1.5 s) | -0.02 to -0.04 per frame |
| Jump spam while stuck | -0.15 per jump after the 2nd |
| Stuck cap | at most -4 per stuck stretch, for the two rows above |
| More than 6 tiles from every route | -0.003 per tile per frame |
| Death | -15 |
| Finish (before anyone finished) | +50 + up to 20 for speed, -0.002 per frame |
| Finish (speed phase) | +50 + 60 x best/own time, +25 for a record, -0.01 per frame |

## Observation (282 inputs)

| Inputs | Count |
|---|---|
| Position, velocity, ground, wall, facing, goal direction / distance | 10 |
| Local map 11x11 | 121 |
| Own route | 6 |
| Moving hazards | 8 |
| Walls | 3 |
| Wind / orbs | 6 |
| Surface map 11x11 | 121 |
| Stuck | 2 |
| Platform | 4 |
| Jump held | 1 |

Older models load: new input columns start small.

## Not verified in the running game yet

- The jump re-press on walls.
- Platform boxes lining up with the drawn platforms.
- What accelerator tiles (id 98) do.
- Whether the lava fill ever closes a real area (pockets over 150 cells stay open).
- Moving saws on tracks are assumed to update their piece position like other movers.

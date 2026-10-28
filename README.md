# Emergence 2

**The next level: a learning AI that lives off the world of [Emergence](https://github.com/anttiluode/Emergence).**

Emergence grew a world from token genomes: cells that eat, share, steal, divide and evolve
into organisms, rings, fronts and territories. Emergence 2 puts a second kind of life on
top of it. **Hunters** are bodies on the grid whose only food is the energy held by living
cells. They have to find it, and they have to learn how, because a hunter that only
wanders randomly barely breaks even.

```
level 2   HUNTERS ─── one shared brain that learns while the world runs
            │  sees a 15x15 patch (cells, cell energy, food, signals, trace, marks, hunters)
            │  conv encoder → GRU memory h → move / write a mark
            │                              → value (future energy)
            │                              → world model (how the patch will change)
            │  bites the cells around it, pays to live and move, splits when rich
            ▼
          trace field  (involuntary smell, fades fast)  ──►  cells perceive it as DANGER
          mark field   (written on purpose, fades slowly) ──►  only hunters read it
            ▼
level 1   CELLS ─── token genomes → transformer → cell behaviour; evolution only
            (unchanged from Emergence, plus 3 new inputs: danger and its gradient)
```

What this sets up, in the order it could happen:

1. **The hunters must model the world to eat.** Organisms, rings, fronts and suns are
   never named to them. If they help to find energy, the brain has a reason to build them.
2. **We can check whether it did.** Level 1 records the answer key: every cell's body,
   clade, age, and where the suns are. `probe.py` asks whether the hunters' memory encodes
   those things, against the raw observation and against an untrained twin brain.
3. **The world can answer back.** Cells now perceive the hunters' trace. Lineages that
   respond to danger (scatter, detach, signal, grow away) can be selected. If the hunters
   keep learning and the cells keep evolving, each is the other's moving target.
4. **The hunters have a medium of their own.** The mark field is a second, slower field
   that hunters choose to write and can read back. It is shared memory if they find a use
   for it (trails, claims, warnings), and wasted energy if they don't.

## Run it

```
pip install -r requirements.txt       # torch with CUDA for your GPU
python run.py                         # 512x512 viewer; cells evolve, hunters arrive at t=1500
python run.py --width 1024 --height 640 --hunters-at 3000
python run.py --load ../Emergence/runs/x/ckpt_0100000.pt   # release hunters into an evolved Emergence world
python run.py --headless --steps 300000 --out runs/night1 --video
python plot.py runs/night1
python probe.py runs/night1/ckpt_0300000.pt --steps 3000   # open the brain
python sanity.py runs/night1/ckpt_0300000.pt --policy brain   # does the learner learn?
```

Viewer: everything from Emergence (views 1–8, zoom box `Z`, wipe, food, inspect), plus
view `9` *hunter fields* (red = trace / danger, blue = marks), `J` releases hunters early
or drops 16 more under the mouse, `L` turns learning on and off live, `X` hides hunters.
Hunters are drawn as small yellow squares, brighter when they have more energy.

Laws are command-line overridable: `--set` for level 1 (as in Emergence), `--hset` for
hunters, e.g. `--hset metabolism=0.25 --hset n_max=1024 --hset learn=0`.

A level-1 checkpoint from Emergence loads with its cell physics unchanged: the first 38
inputs of the cell network are drawn exactly as there, and the 3 danger inputs read zero
until a hunter comes near.

## The hunters, exactly

| | |
|---|---|
| body | one grid site, energy store up to 8 |
| eating | each step, takes 30% of the energy of every living cell in the 3x3 around it (shared fairly if several hunters overlap), keeps 70%; a cell bitten below 0.01 dies |
| costs | 0.20 per step to live, +0.02 to move, +0.01 to write a mark |
| death | at zero energy, or by a small random hazard |
| birth | above energy 5 it splits; the child is born 5–10 cells away with half the energy and an empty memory |
| floor | if fewer than 16 hunters are alive, new ones appear at random (counted as respawns) |
| brain | shared by all hunters: conv encoder on the 15x15x10 patch, GRU(256) memory, heads for policy (5 moves x mark or not), value, and a world model that predicts the change of the next patch and the reward |
| learning | advantage actor-critic on the energy actually gained (rollouts of 16 steps, GAE, entropy bonus 0.03), plus the world-model loss, Adam 5e-4; runs live inside the simulation |

Reward is energy in minus energy spent. Nobody rewards the hunters for anything else.

## What I could check here (CPU only, small worlds)

I had no GPU, so these are short runs on a 128x128 world, one seed each. They test that
the machinery works; they say nothing yet about what a long run on your GPU will grow.

- **The learner learns.** On a frozen world (no regrowth, no deaths, reward = energy
  bitten), 64 hunters with the shared brain bit **0.070–0.087** energy per hunter per step
  in every 400-step window, against **0.053–0.064** for random walkers on the same prey.
  That is roughly 40% more, reached within the first 25 updates. Its policy entropy then
  collapsed to about 0.3, so the default entropy bonus was raised from 0.01 to 0.03.
  With the new defaults the brain kept improving over 4,000 steps, from **0.075** in the
  first window to **0.093** in the last (about 50% above random), while its entropy
  stayed near 2 (still exploring). `sanity.py` reproduces this check.
- **The first ecological trap was real, and fixed.** In the first full runs, children were
  born on their parent's square and copied its policy, so they followed it onto the patch
  it was already eating and starved. The brain did worse than random walkers. Children now
  disperse 5–10 cells.
- **The full ecology oscillates.** With cells regrowing, hunter numbers swing between
  about 15 and 80 on a 128x128 world, with or without learning, like predator–prey
  cycles. Over 2,000–3,000 steps (fewer than 200 brain updates) the learning hunters did
  not yet clearly beat random ones in that setting. That needs the long runs.
- **The world model has not yet beaten "nothing changes".** Its error stayed level with
  the copy-the-last-frame baseline in every short run. It now starts exactly at that
  baseline, so any later drop is learned.
- **The probe runs.** On a barely trained brain, the raw observation predicts the hidden
  quantities better than the memory does, as it should at that stage.

## What to look for on the GPU

| question | where to look |
|---|---|
| Do hunters learn to forage? | `harvest_per_hunter` and `hunters` rising over a run; compare against `--hset policy=random` on the same seed with `plot.py runs/a runs/b` |
| Does their memory model the world? | `probe.py`: hidden quantities where **latent > raw and latent > random** |
| Does the world model learn anything? | `wm_mse` falling below `copy_mse` |
| Do the cells answer back? | over a long run, does harvest per hunter fall while hunters stay smart? Inspect top genomes for changes in detach, divide and signals near danger |
| Do hunters use the mark field? | view 9: do marks form structure (trails, rings around prey, borders) rather than noise? Run `--hset mark_cost=1` to make marks unaffordable and compare harvest |
| Predator–prey cycles | `hunters` and `population` out of phase over time |

## Prior work (fences, not novelty claims)

Predator–prey co-evolution in artificial life (Lotka–Volterra ecologies, Avida, Polyworld);
multi-agent reinforcement learning in grid worlds (Neural MMO, MAgent); world-model agents
(the Dreamer line); probing learned representations against ground truth (Othello-GPT);
and the thermodynamic view that useful memory is predictive (Still et al. 2012). What is
put together here is a learning agent whose only food is an evolving token-genome
ecosystem, which can evolve responses to it, with the ecosystem's hidden structure kept
as an answer key for probing the agent. Whether that combination is new I have not settled.

## Not claimed

- That the hunters understand anything. The probe measures linear readability, not use;
  the intervention test (edit h, watch behaviour change) is the next step and is not built.
- That learning beats random walking in the full regrowing ecology. Not shown yet.
- That cells have evolved any response to danger. The input exists; selection has not
  been observed.
- Speed on GPU. Not measured; the hunters add one small network forward per step and one
  training step every 16.

## Files

```
emergence/          level 1 (from Emergence) + hunters
  hunters.py        level 2: bodies, eating, fields, the brain and its learning
  sim.py            both levels together, drawing, save/load
run.py              viewer and headless runner
probe.py            hidden-variable probes: latent vs raw vs untrained twin
sanity.py           frozen-world check that the learner learns
plot.py             metrics over time, several runs overlaid
inspect_world.py    level-1 genomes, bodies, ancestry
```

# Resolution and adaptation control results

Experiment identifier: `0708f017730c31a8`.

Measurements on the same 104 prototype fields:

- source checkpoint: mAP@50 0.3722 at 640 and 0.5015 at 960;
- adapted-640 checkpoint: 0.9679 at 640 and 0.9704 at 960;
- deployed adapted-960 checkpoint: 0.9648 at 640 and 0.9823 at 960.

Increasing evaluation size alone improved the source checkpoint by 0.1293. At a
fixed 640-pixel evaluation size, adaptation improved mAP@50 by 0.5957. The
historical article values 0.371 and 0.982 come from the original run and are not
substituted by this control. The comparison measures the complete adaptation
recipe; it does not isolate prototype data from replay, augmentation or added
optimization. The 640- and 960-pixel checkpoints are separate runs, so their
difference cannot be attributed exclusively to training resolution.

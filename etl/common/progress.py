"""Per-HUC8 progress markers, so a stage stopped by the Slurm time limit
resumes where it left off.

A stage that loops over HUC8s marks each one when it finishes:

    data/processed/_progress/<case_study>/<stage>/<huc8>

and a re-run skips marked HUC8s. HUC8s a stage skips on purpose (e.g. no
NHDPlus HR package covers them) are recorded under <stage>/skipped/, so
later stages leave them out instead of failing. run_pipeline.sh deletes a stage's markers
when it submits the stage, so a new submission recomputes everything,
while the extra attempts it queues (after a time-out) resume.
"""

from __future__ import annotations


class HucProgress:
    def __init__(self, cfg, stage):
        self.folder = cfg.project_root / "data" / "processed" / "_progress" / cfg.name / stage
        self.folder.mkdir(parents=True, exist_ok=True)

    def done(self, huc8):
        return (self.folder / str(huc8)).exists()

    def mark(self, huc8):
        (self.folder / str(huc8)).touch()

    def skip(self, huc8, reason):
        (self.folder / "skipped").mkdir(exist_ok=True)
        (self.folder / "skipped" / str(huc8)).write_text(str(reason))

    def skipped(self):
        folder = self.folder / "skipped"
        return sorted(p.name for p in folder.iterdir()) if folder.exists() else []

    def clear(self, huc8s):
        """Forget these HUC8s, so they are recomputed (and re-marked)."""
        for huc8 in huc8s:
            (self.folder / str(huc8)).unlink(missing_ok=True)
            (self.folder / "skipped" / str(huc8)).unlink(missing_ok=True)

    def start(self, huc8s, resume):
        """For a stage run over `huc8s`: with resume, the ones already done
        (to skip); without, clear their markers and return []."""
        if not resume:
            self.clear(huc8s)
            return []
        done = [huc8 for huc8 in huc8s if self.done(huc8)]
        if done:
            print(f"Resuming: {len(done)} of {len(huc8s)} HUC8(s) already done "
                  f"(markers in {self.folder})", flush=True)
        return done


def huc8s_with_outputs(cfg):
    """The case study's HUC8s, minus any snap_features / link_dams skipped on
    purpose (e.g. no NHDPlus HR package): those have no outputs, so later
    stages leave them out rather than fail."""
    huc8s = [huc8.zfill(8) for huc8 in cfg.huc8s]
    skipped = set(HucProgress(cfg, "snap_features").skipped()) | set(
        HucProgress(cfg, "link_dams").skipped())
    if skipped:
        print(f"Leaving out {len(skipped)} HUC8(s) skipped by snap_features/link_dams: "
              f"{sorted(skipped)}")
    return [huc8 for huc8 in huc8s if huc8 not in skipped]

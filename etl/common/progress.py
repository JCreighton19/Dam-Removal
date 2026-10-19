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

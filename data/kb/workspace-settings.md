# Workspace Settings

Flowlytics workspaces have a single **workspace timezone** that controls how all
scheduled activity (exports, digest emails, retention windows) is interpreted.

## Changing the workspace timezone

1. Go to **Settings → Workspace → General**.
2. Update the **Timezone** field.
3. Click **Save**.

Changing the workspace timezone does **not** automatically re-save existing
scheduled jobs. Any export, report, or alert that was scheduled using a
specific clock time (e.g. "run at 09:00") keeps its original underlying UTC
offset until the schedule is manually re-saved. This is a common source of
"my export stopped running" or "my export ran at the wrong time" issues after
a timezone change.

## Recommended steps after a timezone change

- Re-open every scheduled export and click **Save** again, even if no fields
  were changed. This forces the schedule to recalculate against the new
  timezone.
- Check **Settings → Workspace → Scheduled Jobs** for a red "stale schedule"
  badge, which flags jobs that still reference the old timezone offset.
- Wait up to 15 minutes for the scheduler cache to refresh after saving.

See also: `scheduled-exports.md` for export-specific troubleshooting.

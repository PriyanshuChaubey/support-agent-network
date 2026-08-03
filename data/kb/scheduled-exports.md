# Scheduled Exports

Scheduled exports run on a per-report basis and push data to the configured
destination (email, S3 bucket, or webhook) at a fixed cadence.

## Why a scheduled export might stop running

1. **Stale schedule after a timezone change.** If the workspace timezone was
   changed after the export was scheduled, the job may silently stop firing
   until it is re-saved. See `workspace-settings.md`.
2. **Destination authentication expired.** S3 and webhook destinations use
   credentials that can expire or be revoked. Check
   **Settings → Integrations → Export Destinations** for a red "needs
   re-auth" indicator.
3. **Report deleted or archived.** If the underlying report was archived, its
   export schedule is automatically paused, not deleted.
4. **Export quota reached.** Free and Starter plans allow 50 scheduled export
   runs per month. Once the quota is hit, remaining runs are skipped until
   the next billing cycle.

## Checklist before escalating an export issue

- Confirm the workspace timezone was recently changed.
- Re-save the schedule and wait 15 minutes.
- Check the destination's authentication status.
- Check **Settings → Workspace → Usage** for export quota.
- Note the exact report name and the last successful run timestamp.

# Troubleshooting Exports

## Export runs but the file is empty

Usually caused by a filter that excludes all rows after a schema change.
Re-check the report's filter conditions.

## Export fails with a permissions error

The destination integration (S3 / webhook) credential has likely expired.
Re-authenticate under **Settings → Integrations → Export Destinations**.

## Export is slow or times out

Reports over 500k rows may take longer than the default 10-minute export
window on Starter plans. Consider narrowing the date range or upgrading to
a plan with an extended export window.

## Export stopped running entirely

See `scheduled-exports.md` and `workspace-settings.md` — this is most often
caused by a workspace timezone change that left the schedule stale.

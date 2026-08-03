# API Credentials

API credentials authenticate requests to the Flowlytics REST API.

## Creating a key

1. Go to **Settings → API → Credentials**.
2. Click **Create Key**.
3. Choose a scope: `full`, `read-only`, or `export-only`.
4. Copy the key immediately — it is shown only once.

## Who can create keys

Only **Owner** and **Admin** roles can create, rotate, or revoke API keys.
This is enforced regardless of individual permission overrides. See
`user-roles-permissions.md` for the full role matrix.

## Key scopes

- `full` — read and write access to all workspace resources.
- `read-only` — matches what a read-only user could see in the UI.
- `export-only` — can trigger and download exports, nothing else.

## Rotation policy

Keys do not expire automatically, but Flowlytics recommends rotating keys
every 90 days. Revoked keys stop working within 60 seconds.

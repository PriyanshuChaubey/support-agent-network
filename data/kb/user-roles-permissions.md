# User Roles and Permissions

Flowlytics has four workspace roles:

| Role       | Can view data | Can edit reports | Can manage users | Can create API credentials |
|------------|---------------|-------------------|-------------------|------------------------------|
| Owner      | Yes           | Yes               | Yes               | Yes                          |
| Admin      | Yes           | Yes               | Yes               | Yes                          |
| Editor     | Yes           | Yes               | No                | No                            |
| Read-only  | Yes           | No                | No                | No                            |

## API credentials and the read-only role

Read-only users can view dashboards, reports, and exports, but **cannot**
create, rotate, or revoke API credentials, regardless of any other
permission grant. API credential management is restricted to the Owner and
Admin roles only, because API keys inherit the permission level of the
workspace rather than the individual user.

If a read-only user needs programmatic access, an Admin or Owner must
generate a **scoped API key** on the read-only user's behalf from
**Settings → API → Credentials → Create Key**, and select "Read-only scope."

See also: `api-credentials.md`.

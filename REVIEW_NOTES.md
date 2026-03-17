# KRN MRP review notes

## Critical fixes applied
- Fixed an import-time migration bug in `migrate_schema()` that could crash the app before startup on a fresh database.
- Removed hardcoded production database credentials from `render.yaml`.
- Added `SESSION_SECRET` as a generated environment variable in Render.
- Externalized user passwords to environment variables while preserving current defaults for backward compatibility.

## Important go-live notes
- The codebase currently mixes generic SQLAlchemy ORM with several PostgreSQL-specific raw SQL queries (`::float`, `jsonb`, `public.` schema, `regexp_replace`, etc.).
- Because of that, the app should be treated as **PostgreSQL-first**. The current README claim that SQLite is the default local path is not operationally true for multiple routes.
- Before go-live, deploy on Postgres/Neon and run end-to-end data entry tests for GRN → Melting → Atomization → Annealing → Grinding → FG → Dispatch.

## Recommended next fixes (not applied here)
1. Remove PostgreSQL-only SQL from routes if SQLite local support is required.
2. Move credentials out of code completely and add a user master table.
3. Add transaction rollback + friendly UI error messages on all POST routes.
4. Add unique validations on generated document numbers under concurrency.
5. Add stock reservation logic before dispatch creation.

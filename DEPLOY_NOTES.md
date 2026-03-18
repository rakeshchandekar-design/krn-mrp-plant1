# Clean deploy notes

This package removes local SQLite fallback from the live app.

Deploy steps:
1. Replace the GitHub repo contents with this package.
2. In Render, do **Manual Deploy → Clear build cache & deploy**.
3. In Neon production, run:
   DROP SCHEMA public CASCADE;
   CREATE SCHEMA public;
4. Open `/setup` once.

If old data still appears after these steps, the running service is still connected to a different database/branch than the one being reset.

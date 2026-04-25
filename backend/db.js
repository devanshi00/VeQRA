import pkg from "pg";
const { Pool } = pkg;

/**
 * PostgreSQL connection pool
 * - Uses node-postgres Pool to manage connections efficiently.
 * - For local development credentials are hard-coded here, but in production
 *   prefer using environment variables (process.env.DB_USER, etc.) for security.
 * - The pool handles connection reuse and simple concurrency for queries made
 *   throughout the backend (import this `pool` and call pool.query(...)).
 */
const pool = new Pool({
  user: "kartikey",
  host: "localhost",
  database: "isro_gi",
  password: "isro_gi",
  port: 5432,
});

/**
 * Export the configured pool instance.
 * - Consumers should `import pool from "./db.js"` and use pool.query(...) or
 *   acquire/release clients for transactions.
 */
export default pool;

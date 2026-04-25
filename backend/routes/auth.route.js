import express from "express";
import bcrypt from "bcrypt";
import pool from "../db.js";

const router = express.Router();

/**
 * auth.route.js
 * - Authentication endpoints for the backend API.
 * - Provides:
 *   - POST /signup : create a new user (hashes password with bcrypt)
 *   - POST /login  : verify credentials and return basic user info
 *
 * Notes:
 * - Passwords are never returned to the client.
 * - This module uses the DB pool for simple queries; in production consider
 *   adding rate limiting, stronger validation and more informative audit logs.
 */

/* -------------------------------------------------------------------------- */
/* POST /api/auth/signup                                                      */
/* - Expects { name, email, password } in the request body.                   */
/* - Performs server-side validation, hashes the password with bcrypt,        */
/*   inserts a new user row and returns the inserted user record (without pw).*/
/* - Handles duplicate email (unique constraint) and general DB errors.       */
/* -------------------------------------------------------------------------- */
router.post("/signup", async (req, res) => {
  const { name, email, password } = req.body;

  // Basic required-field validation
  if (!name || !email || !password)
    return res.status(400).json({ error: "All fields required" });

  try {
    // Hash the plain-text password before storing (bcrypt with 10 salt rounds)
    const hashedPassword = await bcrypt.hash(password, 10);

    // Insert the new user and return id/name/email only (never return password)
    const result = await pool.query(
      "INSERT INTO users (name, email, password) VALUES ($1, $2, $3) RETURNING id, name, email",
      [name, email, hashedPassword]
    );

    // Success: send back the created user's basic info
    res.json(result.rows[0]);
  } catch (err) {
    // Log server-side, but return a user-friendly message
    console.error(err);
    // PostgreSQL unique-violation code 23505 = duplicate key (email already exists)
    if (err.code === "23505") res.status(400).json({ error: "Email exists" });
    else res.status(500).json({ error: "Signup failed" });
  }
});

/* -------------------------------------------------------------------------- */
/* POST /api/auth/login                                                       */
/* - Expects { email, password } in the request body.                         */
/* - Looks up the user by email, compares hashed password using bcrypt.compare*/
/* - On success returns minimal user info (id, name, email).                  */
/* - On failure returns a 400 with an appropriate message.                    */
/* -------------------------------------------------------------------------- */
router.post("/login", async (req, res) => {
  const { email, password } = req.body;

  // Validate presence of credentials
  if (!email || !password)
    return res.status(400).json({ error: "Email and password required" });

  // Fetch user by email
  const result = await pool.query("SELECT * FROM users WHERE email=$1", [
    email,
  ]);
  if (result.rows.length === 0)
    return res.status(400).json({ error: "User not found" });

  const user = result.rows[0];

  // Compare supplied password with stored bcrypt hash
  const match = await bcrypt.compare(password, user.password);
  if (!match) return res.status(400).json({ error: "Invalid password" });

  // Successful authentication: return non-sensitive user data
  res.json({ id: user.id, name: user.name, email: user.email });
});

export default router;

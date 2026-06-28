//! SQLite-backed persistence of per-request records.
//!
//! Records are the durable audit trail behind the live Prometheus counters:
//! every proxied inference is written here so `/stats` and offline analysis can
//! reconstruct exactly what was measured.

use std::sync::Mutex;

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// One persisted inference measurement.
#[derive(Debug, Clone, Serialize)]
pub struct RequestRecord {
    pub ts: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub latency_ms: u64,
    pub energy_j: f64,
    pub electricity_wh: f64,
    pub co2_g: f64,
    pub cost_usd: f64,
    pub status: u16,
    pub streamed: bool,
    pub token_source: String,
}

impl RequestRecord {
    /// Stamp a record with the current UTC time in RFC3339.
    pub fn now() -> String {
        OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_else(|_| String::from("1970-01-01T00:00:00Z"))
    }
}

/// Thread-safe handle to the SQLite request log.
pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Open (creating if needed) the database at `path` and ensure the schema.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS requests (
                 id             INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts             TEXT    NOT NULL,
                 model          TEXT    NOT NULL,
                 input_tokens   INTEGER NOT NULL,
                 output_tokens  INTEGER NOT NULL,
                 latency_ms     INTEGER NOT NULL,
                 energy_j       REAL    NOT NULL,
                 electricity_wh REAL    NOT NULL,
                 co2_g          REAL    NOT NULL,
                 cost_usd       REAL    NOT NULL,
                 status         INTEGER NOT NULL,
                 streamed       INTEGER NOT NULL,
                 token_source   TEXT    NOT NULL
             );",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Persist one record. Failures are returned, never panicked, so a logging
    /// problem cannot take down request handling.
    pub fn record(&self, r: &RequestRecord) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "INSERT INTO requests
                (ts, model, input_tokens, output_tokens, latency_ms,
                 energy_j, electricity_wh, co2_g, cost_usd, status, streamed, token_source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![
                r.ts,
                r.model,
                r.input_tokens,
                r.output_tokens,
                r.latency_ms,
                r.energy_j,
                r.electricity_wh,
                r.co2_g,
                r.cost_usd,
                r.status,
                r.streamed as i64,
                r.token_source,
            ],
        )?;
        Ok(())
    }

    /// Return the most recent `limit` records, newest first.
    pub fn recent(&self, limit: u32) -> Result<Vec<RequestRecord>> {
        let conn = self.conn.lock().expect("store mutex");
        let mut stmt = conn.prepare(
            "SELECT ts, model, input_tokens, output_tokens, latency_ms,
                    energy_j, electricity_wh, co2_g, cost_usd, status, streamed, token_source
             FROM requests ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit], |row| {
            Ok(RequestRecord {
                ts: row.get(0)?,
                model: row.get(1)?,
                input_tokens: row.get::<_, i64>(2)? as u64,
                output_tokens: row.get::<_, i64>(3)? as u64,
                latency_ms: row.get::<_, i64>(4)? as u64,
                energy_j: row.get(5)?,
                electricity_wh: row.get(6)?,
                co2_g: row.get(7)?,
                cost_usd: row.get(8)?,
                status: row.get::<_, i64>(9)? as u16,
                streamed: row.get::<_, i64>(10)? != 0,
                token_source: row.get(11)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Aggregate totals across all recorded requests.
    pub fn totals(&self) -> Result<Totals> {
        let conn = self.conn.lock().expect("store mutex");
        conn.query_row(
            "SELECT COUNT(*),
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(energy_j), 0),
                    COALESCE(SUM(co2_g), 0),
                    COALESCE(SUM(cost_usd), 0)
             FROM requests",
            [],
            |row| {
                Ok(Totals {
                    requests: row.get::<_, i64>(0)? as u64,
                    input_tokens: row.get::<_, i64>(1)? as u64,
                    output_tokens: row.get::<_, i64>(2)? as u64,
                    energy_j: row.get(3)?,
                    co2_g: row.get(4)?,
                    cost_usd: row.get(5)?,
                })
            },
        )
        .map_err(Into::into)
    }
}

/// Lifetime aggregate totals for the `/stats` summary.
#[derive(Debug, Clone, Serialize)]
pub struct Totals {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub energy_j: f64,
    pub co2_g: f64,
    pub cost_usd: f64,
}

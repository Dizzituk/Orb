# Purpose: Nat's first workload — three memory jobs (keyword trail, enrichment,
#          importance). Each job is a CALLER of the generic Nat worker
#          (app.llm.workers.nat_worker.run_nat_job); none owns Nat, and Nat owns
#          nothing about memory. Every job is failure-proof: errors/timeouts
#          return empty and never break a chat reply.

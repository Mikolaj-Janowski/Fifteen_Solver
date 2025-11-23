@echo off
setlocal
set INPUT=input.txt

rem Create logs folder
if not exist logs mkdir logs

rem 1) BFS with explicit successor order DULR
echo ----------------------------------------
echo Running: --bfs DULR
python fifteen_solver.py --bfs DULR < "%INPUT%" > out_bfs_DULR.txt 2> logs\out_bfs_DULR.err
if errorlevel 1 (
  echo ERROR: BFS DULR failed. See logs\out_bfs_DULR.err
  type logs\out_bfs_DULR.err
) else echo OK: out_bfs_DULR.txt

rem 2) BFS with random per-node successor order (ORDER starts with 'R')
echo ----------------------------------------
echo Running: --bfs R
python fifteen_solver.py --bfs R < "%INPUT%" > out_bfs_random.txt 2> logs\out_bfs_random.err
if errorlevel 1 (
  echo ERROR: BFS random failed. See logs\out_bfs_random.err
  type logs\out_bfs_random.err
) else echo OK: out_bfs_random.txt

rem 3) DFS with successor order LURD
echo ----------------------------------------
echo Running: --dfs LURD
python fifteen_solver.py --dfs LURD < "%INPUT%" > out_dfs_LURD.txt 2> logs\out_dfs_LURD.err
if errorlevel 1 (
  echo ERROR: DFS failed. See logs\out_dfs_LURD.err
  type logs\out_dfs_LURD.err
) else echo OK: out_dfs_LURD.txt

rem 4) IDDFS with successor order DULR, max depth 80
echo ----------------------------------------
echo Running: --idfs DULR --maxdepth 80
python fifteen_solver.py --idfs DULR --maxdepth 80 < "%INPUT%" > out_iddfs.txt 2> logs\out_iddfs.err
if errorlevel 1 (
  echo ERROR: IDDFS failed. See logs\out_iddfs.err
  type logs\out_iddfs.err
) else echo OK: out_iddfs.txt

rem 5) Best-first (greedy) with h=0 (zero heuristic). Note: use --bf (short -g is available but we use long)
echo ----------------------------------------
echo Running: --bf 0 --order DULR
python fifteen_solver.py --bf 0 --order DULR < "%INPUT%" > out_bf_h0.txt 2> logs\out_bf_h0.err
if errorlevel 1 (
  echo ERROR: Best-first h0 failed. See logs\out_bf_h0.err
  type logs\out_bf_h0.err
) else echo OK: out_bf_h0.txt

rem 6) Best-first with misplaced-tile heuristic (id=1)
echo ----------------------------------------
echo Running: --bf 1 --order DULR
python fifteen_solver.py --bf 1 --order DULR < "%INPUT%" > out_bf_h1.txt 2> logs\out_bf_h1.err
if errorlevel 1 (
  echo ERROR: Best-first h1 failed. See logs\out_bf_h1.err
  type logs\out_bf_h1.err
) else echo OK: out_bf_h1.txt

rem 7) Best-first with Manhattan heuristic (id=2)
echo ----------------------------------------
echo Running: --bf 2 --order DULR
python fifteen_solver.py --bf 2 --order DULR < "%INPUT%" > out_bf_h2.txt 2> logs\out_bf_h2.err
if errorlevel 1 (
  echo ERROR: Best-first h2 failed. See logs\out_bf_h2.err
  type logs\out_bf_h2.err
) else echo OK: out_bf_h2.txt

rem 8) A* with zero heuristic
echo ----------------------------------------
echo Running: --astar 0 --order DULR
python fifteen_solver.py --astar 0 --order DULR < "%INPUT%" > out_astar_h0.txt 2> logs\out_astar_h0.err
if errorlevel 1 (
  echo ERROR: A* h0 failed. See logs\out_astar_h0.err
  type logs\out_astar_h0.err
) else echo OK: out_astar_h0.txt

rem 9) A* with misplaced-tile heuristic
echo ----------------------------------------
echo Running: --astar 1 --order DULR
python fifteen_solver.py --astar 1 --order DULR < "%INPUT%" > out_astar_h1.txt 2> logs\out_astar_h1.err
if errorlevel 1 (
  echo ERROR: A* h1 failed. See logs\out_astar_h1.err
  type logs\out_astar_h1.err
) else echo OK: out_astar_h1.txt

rem 10) A* with Manhattan heuristic
echo ----------------------------------------
echo Running: --astar 2 --order DULR
python fifteen_solver.py --astar 2 --order DULR < "%INPUT%" > out_astar_h2.txt 2> logs\out_astar_h2.err
if errorlevel 1 (
  echo ERROR: A* h2 failed. See logs\out_astar_h2.err
  type logs\out_astar_h2.err
) else echo OK: out_astar_h2.txt

rem 11) SMA* (textbook) with Manhattan heuristic (default memory)
echo ----------------------------------------
echo Running: --sma 2 --order DULR
python fifteen_solver.py --sma 2 --order DULR < "%INPUT%" > out_sma_defaultmem.txt 2> logs\out_sma_defaultmem.err
if errorlevel 1 (
  echo ERROR: SMA* default mem failed. See logs\out_sma_defaultmem.err
  type logs\out_sma_defaultmem.err
) else echo OK: out_sma_defaultmem.txt

rem 12) SMA* with explicit memory bound 10000
echo ----------------------------------------
echo Running: --sma 2 --order DULR --mem 10000
python fifteen_solver.py --sma 2 --order DULR --mem 10000 < "%INPUT%" > out_sma_mem10000.txt 2> logs\out_sma_mem10000.err
if errorlevel 1 (
  echo ERROR: SMA* mem10000 failed. See logs\out_sma_mem10000.err
  type logs\out_sma_mem10000.err
) else echo OK: out_sma_mem10000.txt

rem 13) Produce a canonical solution file using A* (Manhattan) saved to solution.txt
echo ----------------------------------------
echo Producing solution.txt using: --astar 2 --order DULR
python fifteen_solver.py --astar 2 --order DULR < "%INPUT%" > solution.txt 2> logs\solution.err
if errorlevel 1 (
  echo ERROR: producing solution.txt failed. See logs\solution.err
  type logs\solution.err
  echo Skipping viewer steps.
  goto :end
) else echo OK: solution.txt created.

rem 14) Extract second line (move string) from solution.txt safely and run viewer if available
echo ----------------------------------------
set "MOVES="
for /f "usebackq skip=1 delims=" %%a in ("solution.txt") do (
  set "MOVES=%%a"
  goto :have_moves
)
:have_moves
if "%MOVES%"=="" (
  echo solution.txt has no moves on line 2 or is empty. Skipping built-in viewer run.
) else (
  echo Running built-in viewer with moves: %MOVES%
  python fifteen_solver.py --view "%MOVES%" < "%INPUT%" > viewer_replay_log.txt 2> logs\viewer_replay.err
  if errorlevel 1 (
    echo ERROR: viewer run failed. See logs\viewer_replay.err
    type logs\viewer_replay.err
  ) else echo OK: viewer_replay_log.txt
)

rem 15) Direct viewer example with a literal move string
echo ----------------------------------------
echo Running built-in viewer literal example (--view "LRUDLR")
python fifteen_solver.py --view "LRUDLR" < "%INPUT%" > viewer_direct.txt 2> logs\viewer_direct.err
if errorlevel 1 (
  echo ERROR: viewer direct run failed. See logs\viewer_direct.err
  type logs\viewer_direct.err
) else echo OK: viewer_direct.txt

:end
echo ----------------------------------------
echo ALL COMMANDS FINISHED.
echo Logs are in the \"logs\" folder.
pause
endlocal
exit /b


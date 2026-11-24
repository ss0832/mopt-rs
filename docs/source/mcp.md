# MCP server

mopt-rs offers its commands and some of its components to agents through the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP): a server over standard input and
output that an MCP client (an assistant, an IDE, an agent framework) starts as a child process.

```console
mopt mcp --root /path/to/project               # the Rust binary: the calculators built into Rust
python -P -m mopt_rs.mcp --root /path/to/project   # the Python package: also PySCF, tblite, ...
mopt-mcp --root /path/to/project               # the same, as a command of the Python package
```

Options of both: `--root DIR` (the only directory the server reads and writes for requests;
default: the working directory), `--max-jobs N` (jobs that may run at a time; default 4),
`--allow-external` (allow settings that start programs outside mopt-rs or load code, such as
ASE calculators, external programs and `hook=`; refused by default, see Safety below),
`--full-schemas` (list every option of every command in `tools/list`; see Tools) and
`--page-size N` (entries per page of the lists; default 100).

Both servers offer the same tools (with the same definitions: a test compares them), resources
and prompts. The Rust binary evaluates only the potentials built into Rust (`lj`, `emt`, `eam`,
`meam`, `tersoff`, `muller_brown`, `gfn0xtb`, `gfn1xtb`, `gfnff`); the server of the Python
package runs its jobs through the pip-installed commands, so that every engine of
`mopt_rs.calculators` is available. The component tools that compute from structures are the
same code in both (the Python server calls them through `mopt_rs._core.mcp_component`).

## Connecting a client

A client starts the server with a command line. In the configuration format that most MCP
clients read (a JSON object `mcpServers`):

```json
{
  "mcpServers": {
    "mopt-rs": {
      "command": "python",
      "args": ["-P", "-m", "mopt_rs.mcp", "--root", "/path/to/project"]
    }
  }
}
```

or, for the Rust binary, `"command": "mopt"` with `"args": ["mcp", "--root", "/path/to/project"]`.
Add `"--allow-external"` to the arguments to run ASE calculators, external programs, or code of
your own through the server.

## Protocol

The server follows the specification revision `2026-07-28`
([specification](https://modelcontextprotocol.io/specification/2026-07-28)) and the revisions
with the `initialize` handshake before it (`2025-11-25`, `2025-06-18`, `2025-03-26`,
`2024-11-05`), as a "dual-era" server in the words of the
[versioning page](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning):

- A request whose `params._meta` carries `io.modelcontextprotocol/protocolVersion` (and
  `io.modelcontextprotocol/clientCapabilities`) is answered statelessly in that revision; with
  `2026-07-28`, results carry `resultType: "complete"`, the server identity in
  `_meta["io.modelcontextprotocol/serverInfo"]`, and the lists `ttlMs` and `cacheScope`.
  `server/discover` reports the supported versions and capabilities. An unknown version is
  answered with `-32022` (`UnsupportedProtocolVersion`) and the list of supported versions.
- `initialize` negotiates one of the older revisions for the rest of the process (the requested
  one when supported, else `2025-11-25`); `notifications/initialized` and `ping` work as there.
- Methods: `tools/list`, `tools/call`, `resources/list`, `resources/read`,
  `resources/templates/list`, `prompts/list`, `prompts/get`; the tasks (below): `tasks/get`,
  `tasks/cancel`, `tasks/update` (`2026-07-28`) and `tasks/get`, `tasks/result`, `tasks/list`,
  `tasks/cancel` (`2025-11-25`); subscriptions (below): `subscriptions/listen` (`2026-07-28`),
  `resources/subscribe` and `resources/unsubscribe` (the older revisions).
- Pages: the four lists (and `tasks/list`) come in pages of `--page-size` entries with an
  opaque `nextCursor`; the server accepts only cursors it gave out (any other is `-32602`).
- Notifications from the server: `notifications/progress` (to requests that carry
  `_meta.progressToken` and wait: `job_wait`), `notifications/resources/updated`,
  `notifications/subscriptions/acknowledged`, `notifications/tasks` (`2026-07-28`) and
  `notifications/tasks/status` (`2025-11-25`). From the client: `notifications/cancelled` ends
  a waiting request (`job_wait`, `tasks/result`) without an answer, and ends a
  `subscriptions/listen` stream.
- Batches are refused with `-32600`; a resource that does not exist is `-32602` (`-32002` in the
  older revisions).
- Transport: stdio only, one JSON-RPC message per line; requests that wait are answered when
  their job ends, so responses may come in another order than the requests (as JSON-RPC
  allows). Nothing but protocol messages is written to standard output; the server logs to
  standard error. It ends when its input ends; jobs keep running.

## Tools

**One tool per implemented command** (`optmain`, `nebmain`, `ieipmain`, `mdmain`,
`confsearch`, `relaxedscan`, `orientsearch`, `run_autots`, `run_mapper`, `run_rcmc`,
`freqmain`, `ircmain`, `run_kinetics`, `run_automecp`, `etmain`). The tools are generated when
the server starts from the option tables of the command line (`crates/mopt-rs/src/cli/spec.rs`,
written by `tools/cli/gen_cli.py` together with the table of positional inputs,
`POSITIONALS`), so they cannot differ from it: a command that becomes implemented in the tables
is a tool without any change to the server. The arguments of a tool are

- the options of the command line by their long names (the keys of
  [settings files](running.md), e.g. `opt_method`, `othersoft`, `NSTEP`), with the JSON type from
  their arity (a flag: `true`/`false`; one value: a string or a number; several values: an
  array; an optional value: a string, a number, or `true` for the flag alone). Any flag
  spelling (`opt`, `-opt`) is accepted as well; options that are not implemented yet are
  refused with the reason, and an unknown name is refused with a pointer to `command_help`;
- `INPUT`: the positional inputs (paths relative to the root);
- `files`: further files or directories that options name (an initial Hessian, a product
  structure), copied with the inputs.

`tools/list` gives each command tool in a **compact form** by default: `INPUT`, `files` and the
options of `othersoft`, `opt_method`, `NSTEP`, `electronic_charge`, `spin_multiplicity` and
`calc_option` that the command has, with every other option accepted by its long name
(`additionalProperties`). The 37 tools take 60 kB of JSON so, against 479 kB with every
option (clients keep tool schemas in the context of a model). `command_help` gives the rest:
the `--help` text of a command, the schema and help of the options that `options` names (long
names, flags, or parts of them: `["-ns", "modelhess"]`), and with `schema: true` the full input
schema of the tool. `--full-schemas` lists the full schemas instead.

The tool starts the command as a job and answers at once with the `job_id` and the run
directory (see below), or with a task for a client that asks for tasks (see Tasks).

**Job tools.** `job_status` (state, exit code, run directories, the tail of `mopt.log`, the
last iteration printed there, the tail of the console output), `job_wait` (waits up to
`timeout_seconds`, at most 600, until the job does not run, then answers as `job_status` with
`waited_seconds` and `timed_out`; with `_meta.progressToken` the request gets
`notifications/progress` at every new iteration in `mopt.log`, and `notifications/cancelled`
ends it without an answer), `job_stop` (a `STOP` file: the command ends cleanly at its next
check), `job_cancel` (the process of a job of this server process is ended at once; state
`cancelled`), `job_restart`, `job_result` (the files of the run directories with sizes and
resource URIs, and the totals of the evaluations), `job_list`, and `mapper_steer` (lines of
`steer.txt` for a running `run_mapper` job, see [mapper.md](mapper.md): `explore NAME`,
`skip NAME`, `unskip NAME`, `max_jobs N`, checked by their form; `steer.log` tells what became
of them).

**Component tools**, answered at once: `list_commands` (the commands, whether they are
implemented, their numbers of options, their tools), `command_help` (above),
`list_calculators` (the calculators of the registry and the objectives of plain variables),
`single_point` (energy in Hartree and gradient in Hartree/Bohr of one xyz structure,
optionally the Hessian), and the tools that compute from structures given as files under the
root or as xyz text:

| Tool | What it answers |
|---|---|
| `model_hessian` | a model Hessian as `-modelhess` gives it (any name, `legacy_` and the modifiers; `gfn1xtb`: the analytic GFN1-xTB Hessian), with its lowest eigenvalues |
| `vibrations` | normal modes and thermochemistry (as `freqmain`) from a given Hessian or from the Hessian of a calculator |
| `coordinates` | the coordinates of a coordinate system of `-coord` at a structure: their number, labels, values |
| `interpolate` | the first images between two structures (`linear`, `lst`, `idpp`, `sidpp`, `geodesic`, `fbenm`, `cfbenm`) as a multi-frame xyz text |
| `superpose` | the RMSD (Angstrom) after Kabsch superposition, and the moved structure |
| `bond_graph` | bonds, fragments, and the numbers of angles and dihedrals |
| `dispersion` | the D3(BJ) or D4 energy and gradient (PBE parameters) |
| `hessian_update` | one quasi-Newton update of a given Hessian (`bfgs`, `sr1`, `psb`, `bofill`, `fsb`, `msp`, `ts_bfgs`, `flowchart`, the `block_` forms) |
| `units` | the conversion factors, and a value converted between energy or length units |
| `elements` | atomic number, symbol, mass and covalent radius of elements |

Structures of these tools have at most 2000 atoms (300 where the answer is a Hessian).

Failures that the caller can correct (an unknown option, a path outside the root, a job that is
not running) are tool results with `isError: true`; an unknown tool is the protocol error
`-32602`.

An example exchange (one line each, shortened):

```text
-> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"optmain","arguments":
    {"INPUT":["ar7.xyz"],"othersoft":"lj","opt_method":["rsirfo_bfgs"],"NSTEP":300}}}
<- {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"..."}],"isError":false,
    "structuredContent":{"job_id":"job-5f0c...e41a","state":"running",
    "run_directory":"mcp_jobs/job-5f0c...e41a/2026_11_17/ar7_OPT_ASE_...", ...}}}
-> {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"job_status","arguments":
    {"job_id":"job-5f0c...e41a"}}}
<- {... "structuredContent":{"state":"finished","exit_code":0,"log_tail":"...converged!!!...", ...}}
```

## Jobs

A job is the command run as a child process (the same `mopt` binary, or the pip-installed
command for the Python server), so that a failing run does not end the server. It uses only
what every command has, and no command needs to know about MCP:

- The job directory is `mcp_jobs/JOB_ID/` under the root. The inputs and the `files` are copied
  into it at their paths relative to the root, so that every relative path of an option means
  the same file there. The arguments of the tool are written to `mcp_settings.json` and the
  command runs with `--config mcp_settings.json` (exactly the settings file of the command line);
  it writes its run directory there as usual (`settings.json`, `mopt.log`, the evaluation
  journal, its results). The console output goes to `console.log`.
- **Stop**: `job_stop` writes `STOP` into the run directory; the command ends cleanly at its
  next check (exit code 3, state `interrupted`).
- **Restart**: `job_restart` runs the command with `--restart RUN_DIR` in the job directory,
  which continues from the settings and the evaluation journal kept in the run directory.
- **Result**: `job_result` lists the files of the run directory (logs, CSV tables, trajectories,
  HTML reports, plots, and whatever else the command writes); they are read with
  `resources/read`.
- **Cancel**: `job_cancel` ends the process at once (the Python server ends the process group
  of the job, so that workers an engine started end with it); state `cancelled`. What the run
  wrote stays, and `job_restart` continues it from its journal. Only the jobs that this server
  process started can be ended so; `job_stop` asks the others.
- States: `running`, `finished` (exit code 0), `interrupted` (3), `failed` (other codes),
  `cancelled` (ended by `job_cancel`), `unknown` (started by another server process and not
  seen running). The child records its exit code in `mcp_exit_code` and the server what it
  started (and which attempt was cancelled) in `mcp_job.json`, so that a server started later
  knows how a job ended; jobs continue when the server exits.

## Tasks

A command tool runs as a task for a client that asks for one; the task is the job (its
`taskId` is the `job_id`), and the job tools work on it as well.

- `2026-07-28`: the extension `io.modelcontextprotocol/tasks` (declared by the server in
  `server/discover`, `capabilities.extensions`). A `tools/call` of a command tool whose request
  declares the extension in its client capabilities is answered with a task
  (`resultType: "task"`, `taskId`, `status`, `statusMessage`, `createdAt`, `lastUpdatedAt`,
  `ttlMs: null`, `pollIntervalMs`). `tasks/get` gives the task with `result` (the result of
  `job_result`) once it is `completed` and `error` once it `failed`; `tasks/cancel` cancels it;
  `tasks/update` is refused (the jobs ask no questions); `notifications/tasks` come through
  `subscriptions/listen` with `taskIds`.
- `2025-11-25`: the experimental tasks (capability `tasks` with `list`, `cancel` and
  `requests.tools.call`; the command tools carry `execution.taskSupport: "optional"`). A
  `tools/call` with `task` is answered with `{"task": {...}}` (`ttl`, `pollInterval`);
  `tasks/get`, `tasks/result` (waits until the task ends, then the result of `job_result`, with
  `_meta["io.modelcontextprotocol/related-task"]`), `tasks/list` (in pages), `tasks/cancel`,
  and `notifications/tasks/status` for the tasks made in the session.

The status follows the latest attempt of the job: running is `working`, finished `completed`,
interrupted and cancelled `cancelled`, failed and unknown `failed`. Cancelling a task is
cooperative: the job gets `STOP` and ends cleanly at its next check (the task is `cancelled`
at once and stays so; `job_restart` can still continue the job). Tasks are kept as long as
their jobs.

## Subscriptions

A resource can be subscribed to: a file or directory under the root (it may not exist yet: a
log that a run will write), a documentation page, or the status of a job (`mopt://jobs/JOB_ID`).
The server looks at what is watched every 0.25 s and tells when it changed (its size,
modification time, number of entries, or the state of the job):

- `2026-07-28`: `subscriptions/listen` with `notifications.resourceSubscriptions` (and
  `taskIds`) is acknowledged at once by `notifications/subscriptions/acknowledged` with the
  subscriptions the server agreed to (those it refuses are left out); then
  `notifications/resources/updated` and `notifications/tasks` carry
  `_meta["io.modelcontextprotocol/subscriptionId"]` (the id of the request) until the client
  sends `notifications/cancelled` for it. The lists never change while a server runs, so no
  list-changed notifications are agreed to.
- the older revisions: `resources/subscribe` and `resources/unsubscribe` with `uri`;
  `notifications/resources/updated` with `uri`.

At most 16 streams and 256 watched resources at a time.

## Resources

- `mopt://docs/NAME`: the pages of this documentation. The `mopt` binary embeds them (a test
  checks that every page of `docs/source` is there, with the same text); the Python server
  serves the copy installed with the package.
- `mopt://root/PATH`: a file under the root (a directory gives a listing). `resources/list`
  lists the documentation, the files of the run directories under the root, and the jobs.
  Text files are returned as text, other files (plots) as Base64, up to 8 MiB.
- `mopt://jobs/JOB_ID`: the status of a job as `job_status` gives it (JSON).

## Prompts

`optimize_structure` (an xyz file and a calculator: run `optmain`, follow it, report) and
`follow_job` (follow a job to its end, continue it when it was stopped, summarize its files).

## Safety

The server is meant for an agent that should work inside one directory and run mopt-rs, and
nothing else. Every rule below has a test that attacks it (`crates/mopt-rs/tests/mcp_attacks.rs`,
`tests/test_mcp.py`, `tests/test_mcp_features.py`).

**Paths.** Nothing outside the root is read or written for a request.

- A path from a client (`INPUT`, `files`, `structure`, `run_directory`, resource URIs) is
  refused by its text when it holds a `..` component (also one that would stay inside), a NUL
  character, a backslash, a colon (drives such as `C:`, streams), a reserved device name
  (`CON`, `NUL`, `COM1`, ...), a component ending in a dot or a space, or starts like a UNC path
  (`//server`). Paths use `/` and are relative to the root; an absolute path is accepted only
  when it resolves inside the root.
- The rest is resolved by the operating system (symbolic links and junctions as the system
  follows them) and must lie inside the resolved root. Every file is resolved again when it
  is opened and the opened file is compared with the checked one, so that a link swapped in
  between is noticed; copies are written as new regular files, never through links. Links in
  a copied directory that lead out of the root are skipped (and reported).
- Values of options that may name files (`-ih FILE`, `-calc_opt key=FILE`, `key=@FILE`, the
  strings of a JSON settings file copied in such as `run_autots -cfg`) are seen from the job
  directory: absolute paths, drives, `~`, `..`, backslashes and NUL characters are refused. A
  file an option names is given in `files`, which copies it into the job directory.
- Resource URIs are refused when they encode dots, slashes, backslashes or NUL characters
  (`%2e`, `%2f`, `%5c`, `%00`); they are decoded once. Resources are regular files of at most
  8 MiB; a directory listing names its entries without following links.
- Job ids are tokens the server makes (`job-` and 32 hexadecimal digits). The server finds a
  job by comparing the token with the records under `mcp_jobs/`; a token is never used as a
  path. `mcp_jobs` itself must be a directory inside the root, not a link. Task ids are job ids,
  checked the same way, and so are the ids of `mopt://jobs/` URIs.
- Structure arguments of the component tools are paths under the root, checked and opened as
  inputs are (a file or its text, never both); the tools themselves receive only texts. A
  coordinate system of `coordinates` may take a number as its argument, never a file
  (`hyperspherical:FILE` is refused). `vibrations` with a calculator follows the rules of
  `single_point`.
- Subscriptions are checked as `resources/read` checks a URI when they are made, and the path is
  resolved again at every look: a path that leads outside the root then (a link swapped in) is
  never read, and what happens outside is not told (its fingerprint stays fixed).
- `mapper_steer` writes only `steer.txt` in the run directory of a running `run_mapper` job of
  the root (appended, refused when a link is there), and only lines of the four forms of the
  steering file (no control characters, comments, or other words).
- Cursors are the offset and a check value made with a key of the server process; others are
  refused.

**Programs and code.** Only the commands of mopt-rs are started, as an argument vector,
never through a shell; there is no tool that names another program. Without
`--allow-external` (`mopt mcp --allow-external`, `python -m mopt_rs.mcp --allow-external`),
the server refuses settings that start programs outside mopt-rs or load code:

- calculators other than the built-in potentials, the objectives of plain variables, and the
  libraries `pyscf`, `psi4`, `tblite`, `dxtb` that run inside the job (refused: `ase:module:Class`,
  the ASE profiles, external programs such as ORCA, `multioptpy:NAME`, plugins, OpenMM), in
  `-os` and in the calculators of `-oniom` and `-oniom_middle`;
- references to code `module:name` in any value, and the settings `hook=`, `hook.*`,
  `excited_setup=`, `state_reader=`, `factory=`, `system=` (also as `-oniom_opt low.system=`);
- `-osp` (`software_path_file`), and copied files named `software_path.conf`,
  `mopt_profiles.json` or `mopt_calculator_options.json` (the commands read them from their
  working directory);
- the same in JSON files copied into a job (settings files of commands).

References to code (`module:name`) are looked for in the settings of calculators (`-calc_opt`,
`-oniom_opt`, the settings of a single point), where the engines take them; other options
have `NAME:VALUE` forms of their own (`-coord hyperspherical:FILE`, `dvv:v0`), whose file part
is checked as a path.

The jobs of the Python server run with `python -P`, so that a file copied into a job cannot
stand in for a module. Start the Python server itself with `python -P -m mopt_rs.mcp` when its
working directory holds files that are not yours.

**Limits.** At most `--max-jobs` jobs (default 4) of a server run at a time. A message is at
most 16 MiB and nests arrays and objects at most 64 levels deep; malformed messages are
answered with an error and the server goes on. At most 1 GiB and 20000 files are copied into
one job. A single point and the component tools, which the server computes while the client
waits, take at most 2000 atoms (300 with a Hessian); larger structures are jobs. At most 64
requests wait at a time (`job_wait`, `tasks/result`), each `job_wait` at most 600 s; at most 16
subscription streams and 256 watched resources.

The jobs see the environment of the server.

## What the server does not offer

- **Transports other than stdio.** The server is a child process of its client; an HTTP
  transport would listen on the network, which the rule that nothing outside the root is
  reached for a request does not cover (authentication, origins, other users of the machine).
- **Questions from a job** (`input_required` tasks, `tasks/update` with answers): the commands
  run without questions.
- **List-changed notifications**: the tools, prompts and templates do not change while a server
  runs (the list of resources does, as jobs write files: read it again, or subscribe to the
  files and directories of interest).
- **Component tools for every part of the Python API.** The parts with a short answer are tools
  (above); the parts that run long are reached through the command tools that use them: step
  methods, bias potentials and constraints through `optmain` (`-opt`, `-ma` and the other bias
  options, `-pc`, `-uc`),
  O1NumHess through `-modelhess o1numhess`, orbital analyses and vibrational analyses through
  `freqmain`, the pair methods through `ieipmain`, DMF through `nebmain -DMF`, BITSS through
  `optmain -mf bitss`, IRC through `ircmain`, molecular dynamics and trajectory analyses
  through `mdmain`, conformer and orientation searches through `confsearch` and
  `orientsearch`, ReSPer maps through `relaxedscan`.

Limits of the checks: the checks of option values look at the values the server is given and
at the JSON files copied in. A command may read further paths from other files it is given
(structure files name no paths; a JSON file that names an absolute path, such as a
`reaction_network.json` written elsewhere, is refused as a whole). Programs started with
`--allow-external` are not confined to the root. The Rust server writes its own files inside a
job directory after checking that no link is there; the Python server also opens them with
`O_NOFOLLOW`.

# svaeva-evaluation

<div align="center">

[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/elacosse/svaeva-evaluation/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/charliermarsh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/elacosse/svaeva-evaluation/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/elacosse/svaeva-evaluation/releases)
[![License](https://img.shields.io/github/license/elacosse/svaeva-evaluation)](https://github.com/elacosse/svaeva-evaluation/LICENSE)
![Coverage Report](assets/images/coverage.svg)

`svaeva-evaluation` is a Python cli/package

</div>

## Very first steps

### Prepare your .env file with appropriate parameters and API keys

The following ```.env``` file with the following variables _must_ be placed in the most parent directory of this repository 

```bash
# Redis config
REDIS_HOST=127.0.0.1 # default
REDIS_PORT=6379 # default
REDIS_DB_INDEX=0 # default
REDIS_USER=<YOUR REDIS USER (OPTIONAL)>
REDIS_PASSWORD=<YOUR REDIS PASSWORD (OPTIONAL)>
REDIS_OM_URL=redis://127.0.0.1:6379/0 # Redis URL for redis-om use

# External API keys
OPENAI_API_KEY=<YOUR OPENAI API KEY>
ELEVEN_API_KEY=<YOUR ELEVEN LABS API KEY>

# For svaeva-evaluation
RANDOM_SEED=42
PLATFORM_ID=svaeva-redux
GROUP_ID=consonancia
CONVERSATION_ID=sim-consonancia
```

### Initialize your code

1. If you don't have `Poetry` installed run:

```bash
make poetry-download
```

2. Initialize poetry and install:

```bash
make install

```

## 🚀 Usage
```bash
 Usage: svaeva-evaluation [OPTIONS] COMMAND [ARGS]...                                                                                                                                                         
                                                                                                                                                                                                              
 `svaeva-evaluation` is a Python cli/package to manage the ConsonâncIA installation                                                                                                                           
                                                                                                                                                                                                              
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ display-by-rank              Select the best connected node and save the conversation to a file by rank.                                                                                                   │
│ message                      Message a user by their user-id.                                                                                                                                              │
│ network                      Constuct and save network from all users                                                                                                                                      │
│ plot                         Plot tSNE embeddings, 3D PCA embeddings and edge distribution to data/plots/{group_id}-{platform_id}                                                                          │
│ save-local                   Save images, network and videos locally.                                                                                                                                      │
│ select                       Select the user and do all the great things...                                                                                                                                │
│ version                      Print the version of the package.                                                                                                                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### Display users that are best connected by their rank
```bash
 Usage: svaeva-evaluation display-by-rank [OPTIONS]                                                                                                                                                           
                                                                                                                                                                                                              
 Select the best connected node and save the conversation to a file by rank.                                                                                                                                  
                                                                                                                                                                                                              
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --number  -n      INTEGER  number of users to display [default: 7]                                                                                                                                         │
│ --time    -t      INTEGER  time window in seconds [default: -1]                                                                                                                                            │
│ --help                     Show this message and exit.                                                                                                                                                     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### Select a user and generate assets for room experience.
```bash
 Usage: svaeva-evaluation select [OPTIONS] USER_ID                                                                                                                                                            
                                                                                                                                                                                                              
 Select the user and do all the great things...                                                                                                                                                               
                                                                                                                                                                                                              
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    user_id      TEXT  user-id to select [default: None] [required]                                                                                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                │
╰
```

#### Message a selected user with an invitation (svaeva-platform must be running!)
```bash
 Usage: svaeva-evaluation message [OPTIONS] USER_ID                                                                                                                                                           
                                                                                                                                                                                                              
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    user_id      TEXT  user-id to select [default: None] [required]                                                                                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --message  -m      TEXT  message to send to user.                                                                                                                                                          │
│                          [default: This is Consonância. You're invited to enter the room of healing algorithms for something special. Please type or click with /iamready if you accept this invitation.   │
│                          You have 10 minutes to accept this invitation.]                                                                                                                                   │
│ --help                   Show this message and exit.                                                                                                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

#### Plot embeddings spaces and edge distributions and save to ```data/plots/*```
```bash
 Usage: svaeva-evaluation plot [OPTIONS]                                                                                                                                                                      
                                                                                                                                                                                                              
 Plot tSNE embeddings, 3D PCA embeddings and edge distribution to data/plots/{group_id}-{platform_id}                                                                                                         
                                                                                                                                                                                                              
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --time  -t      INTEGER  time window in seconds [default: -1]                                                                                                                                              │
│ --help                   Show this message and exit.                                                                                                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

###### Sample Edge Distribution
![edge](assets/images/edge_distribution.png)
###### t-SNE Embeddings
![tsne](assets/images/tSNE_embeddings.png)
###### PCA Embeddings 3D Projection
![edge](assets/images/PCA_embeddings.png)

### Generate Sample Conversation Data (For ConsonancIA)

Easily and asynchronously generate sample conversation data to appropriately store in the database to play with.
(Note: svaeva-redux platform must be up and running!)

```bash
python svaeva_evaluation/simulation/consonancia.py
```

### Makefile usage

[`Makefile`](https://github.com/elacosse/svaeva-evaluation/blob/master/Makefile) contains a lot of functions for faster development.

<details>
<summary>1. Download and remove Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks coulb be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>3. Codestyle</summary>
<p>

Automatic formatting uses `pyupgrade` and `ruff`.

```bash
make codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `ruff` and `darglint` library

Update all dev libraries to the latest version using one comand

```bash
make update-dev-deps
```

</p>
</details>

<details>
<summary>4. Code security</summary>
<p>

```bash
make check-safety
```

This command launches `Poetry` integrity checks as well as identifies security issues with `Safety` and `Bandit`.

```bash
make check-safety
```

</p>
</details>

<details>
<summary>5. Type checks</summary>
<p>

Run `mypy` static type checker

```bash
make mypy
```

</p>
</details>

<details>
<summary>6. Tests with coverage badges</summary>
<p>

Run `pytest`

```bash
make test
```

</p>
</details>

<details>
<summary>7. All linters</summary>
<p>

Of course there is a command to ~~rule~~ run all linters in one:

```bash
make lint
```

the same as:

```bash
make test && make check-codestyle && make mypy && make check-safety
```

</p>
</details>

<details>
<summary>8. Docker</summary>
<p>

```bash
make docker-build
```

which is equivalent to:

```bash
make docker-build VERSION=latest
```

Remove docker image with

```bash
make docker-remove
```

More information [about docker](https://github.com/elacosse/svaeva-evaluation/tree/master/docker).

</p>
</details>

<details>
<summary>9. Cleanup</summary>
<p>
Delete pycache files

```bash
make pycache-remove
```

Remove package build

```bash
make build-remove
```

Delete .DS_STORE files

```bash
make dsstore-remove
```

Remove .mypycache

```bash
make mypycache-remove
```

Or to remove all above run:

```bash
make cleanup
```

</p>
</details>

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/elacosse/svaeva-evaluation/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

We use [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.

### List of labels and corresponding titles

|               **Label**               |  **Title in Releases**  |
| :-----------------------------------: | :---------------------: |
|       `enhancement`, `feature`        |       🚀 Features       |
| `bug`, `refactoring`, `bugfix`, `fix` | 🔧 Fixes & Refactoring  |
|       `build`, `ci`, `testing`        | 📦 Build System & CI/CD |
|              `breaking`               |   💥 Breaking Changes   |
|            `documentation`            |    📝 Documentation     |
|            `dependencies`             | ⬆️ Dependencies updates |

You can update it in [`release-drafter.yml`](https://github.com/elacosse/svaeva-evaluation/blob/master/.github/release-drafter.yml).

GitHub creates the `bug`, `enhancement`, and `documentation` labels for you. Dependabot creates the `dependencies` label. Create the remaining labels on the Issues tab of your GitHub repository, when you need them.

## 🛡 License

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/elacosse/svaeva-evaluation/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{svaeva-evaluation,
  author = {Eric Lacosse},
  title = {`svaeva-evaluation` is a Python cli/package},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/elacosse/svaeva-evaluation}}
}
```


# GLHF | Generals.io Bot Framework

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/glhf)
![PyPI - Version](https://img.shields.io/pypi/v/glhf)

Empowering seamless bot-server interactions with customizable workflows, all designed to elevate your automation experience effortlessly.

## Table of Contents

- [GLHF | Generals.io Bot Framework](#glhf--generalsio-bot-framework)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Install from PyPI](#install-from-pypi)
    - [Clone the Repository and Install from Source](#clone-the-repository-and-install-from-source)
  - [Features](#features)
    - [Example Bots](#example-bots)
    - [`glhf.sever`](#glhfsever)
    - [`glhf.gui`](#glhfgui)
    - [`glhf.app`](#glhfapp)
- [Troubleshooting](#troubleshooting)
  - [\[ImportError: DLL load failed while importing \_igraph\]{https://python.igraph.org/en/latest/install.html#q-i-am-trying-to-install-igraph-on-windows-but-am-getting-dll-import-errors}](#importerror-dll-load-failed-while-importing-_igraphhttpspythonigraphorgenlatestinstallhtmlq-i-am-trying-to-install-igraph-on-windows-but-am-getting-dll-import-errors)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

To have examples on your machine, it is recommended to clone the repository.

### Install from PyPI

```sh
pip install glhf
```

### Clone the Repository and Install from Source

```sh
git clone https://github.com/Stanley5249/generals-io-glhf.git
cd generals-io-glhf
pip install .
```

## Features

This project consists of several components, each with its own set of features. Below is a detailed breakdown:

| Status | Type                 |
| :----: | -------------------- |
|   ✅    | Complete             |
|   ⚠️    | Partially Functional |
|   📅    | Planned              |

### Example Bots

| Component        | Status | Description                                           |
| ---------------- | :----: | ----------------------------------------------------- |
| `SurrenderBot`   |   ✅    | ...                                                   |
| `OptimalOpening` |   ✅    | Compute optimal strategy paths for the first 25 turns |
| Human.exe        |   📅    | Plan for future integration                           |

### `glhf.sever`

| Component        | Status | Description                    |
| ---------------- | :----: | ------------------------------ |
| `SocketioServer` |   ✅    | Connect to official site       |
| `LocalServer`    |   ⚠️    | Simulate official site locally |

### `glhf.gui`
| Component   | Status | Description                     |
| ----------- | :----: | ------------------------------- |
| `PygameGUI` |   ⚠️    | Display only map during in-game |


### `glhf.app`
| Component | Status | Description                               |
| --------- | :----: | ----------------------------------------- |
| `APP`     |   ⚠️    | Integrate server, GUI, and bots           |
| `command` |   ⚠️    | Provide command line tool wrapper for APP |

# Troubleshooting

## [ImportError: DLL load failed while importing _igraph]{https://python.igraph.org/en/latest/install.html#q-i-am-trying-to-install-igraph-on-windows-but-am-getting-dll-import-errors}

## Contributing

Your feedback and contributions are highly valued as the project continues to evolve.

## License

This project is licensed under the GNU General Public License Version 2.
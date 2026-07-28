#!/usr/bin/env bash

# ==============================================================================
# SUBTONE: Local Installation and Execution Script for macOS and Linux
# ==============================================================================

set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

log_info() { echo -e "${BLUE}${BOLD}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}${BOLD}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}${BOLD}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
${BOLD}Subtone Utility Script for macOS & Linux${NC}

${BOLD}Usage:${NC}
  ./subtone.sh <command> [arguments...]

${BOLD}Commands:${NC}
  ${CYAN}setup${NC}      Set up a Python virtual environment, install system dependencies,
              and install Subtone in editable mode with development dependencies.
  ${CYAN}test${NC}       Run the test suite using pytest inside the virtual environment.
  ${CYAN}run${NC}        Execute the subtone CLI with passed arguments.
  ${CYAN}shell${NC}      Show instructions on how to activate the virtual environment manually.

${BOLD}Examples:${NC}
  ./subtone.sh setup
  ./subtone.sh test
  ./subtone.sh run /path/to/my/stems --level 5
EOF
}

check_system_dependencies() {
    OS="$(uname -s)"
    log_info "Verifying system environment on ${OS}..."

    PYTHON_BIN=""

    if [[ "${OS}" == "Darwin" ]]; then
        log_success "Verified macOS environment."

        if ! command -v brew &> /dev/null; then
            log_warn "Homebrew is not installed. Native audio libraries might fail to load."
            log_info "Install Homebrew via: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        else
            log_success "Homebrew found."

            if ! brew list libsndfile &> /dev/null; then
                log_info "Installing libsndfile via Homebrew..."
                brew install libsndfile
            fi

            if ! command -v ffmpeg &> /dev/null; then
                log_info "Installing ffmpeg via Homebrew..."
                brew install ffmpeg || log_warn "Failed to install ffmpeg. Run 'brew install ffmpeg' manually."
            fi
        fi

        # Try specific executable candidates in order
        for py in python3.11 python3.12 python3.10 python3; do
            if command -v "$py" &> /dev/null; then
                PYTHON_BIN="$(command -v "$py")"
                break
            fi
        done

    elif [[ "${OS}" == "Linux" ]]; then
        log_success "Verified Linux environment."

        # Install system audio libraries and dependencies based on distro package manager
        if command -v apt-get &> /dev/null; then
            log_info "Debian/Ubuntu system detected."
            export DEBIAN_FRONTEND=noninteractive
            MISSING_PKGS=()
            command -v ffmpeg &> /dev/null || MISSING_PKGS+=("ffmpeg")

            if ! dpkg -s libsndfile1 &> /dev/null && ! dpkg -s libsndfile &> /dev/null; then
                MISSING_PKGS+=("libsndfile1")
            fi

            if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
                log_info "Installing missing system packages: ${MISSING_PKGS[*]}"
                sudo DEBIAN_FRONTEND=noninteractive apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${MISSING_PKGS[@]}"
            fi
        elif command -v dnf &> /dev/null; then
            log_info "Fedora/RHEL system detected."
            MISSING_PKGS=()
            command -v ffmpeg &> /dev/null || MISSING_PKGS+=("ffmpeg")
            rpm -q libsndfile &> /dev/null || MISSING_PKGS+=("libsndfile")

            if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
                log_info "Installing missing system packages: ${MISSING_PKGS[*]}"
                sudo dnf install -y "${MISSING_PKGS[@]}"
            fi
        elif command -v pacman &> /dev/null; then
            log_info "Arch Linux system detected."
            MISSING_PKGS=()
            command -v ffmpeg &> /dev/null || MISSING_PKGS+=("ffmpeg")
            pacman -Qs libsndfile &> /dev/null || MISSING_PKGS+=("libsndfile")

            if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
                log_info "Installing missing system packages: ${MISSING_PKGS[*]}"
                sudo pacman -S --needed --noconfirm "${MISSING_PKGS[@]}"
            fi
        else
            log_warn "Unrecognized Linux package manager. Ensure 'ffmpeg' and 'libsndfile' are installed."
        fi

        # Look for Python 3 binaries
        for py in python3.11 python3.12 python3.10 python3; do
            if command -v "$py" &> /dev/null; then
                PYTHON_BIN="$(command -v "$py")"
                break
            fi
        done

        # If python is found on Debian/Ubuntu, verify venv capability
        if [[ -n "${PYTHON_BIN}" ]] && command -v apt-get &> /dev/null; then
            if ! "${PYTHON_BIN}" -m venv --help &> /dev/null; then
                PY_VER_SHORT="$("${PYTHON_BIN}" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
                log_warn "Virtual environment module missing for ${PY_VER_SHORT}. Installing ${PY_VER_SHORT}-venv..."
                sudo DEBIAN_FRONTEND=noninteractive apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${PY_VER_SHORT}-venv"
            fi
        fi

        # Fallback install if no Python 3 found
        if [[ -z "${PYTHON_BIN}" ]] && command -v apt-get &> /dev/null; then
            log_info "Attempting to install Python 3 via apt..."
            sudo DEBIAN_FRONTEND=noninteractive apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-dev
            PYTHON_BIN="$(command -v python3)"
        fi
    else
        log_warn "Unrecognized OS '${OS}'. Proceeding with standard binary resolution..."
        if command -v python3 &> /dev/null; then
            PYTHON_BIN="$(command -v python3)"
        fi
    fi

    if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
        log_error "Python 3 (>= 3.10) is required but could not be located."
        log_info "Please install Python manually."
        exit 1
    else
        log_success "Using Python executable: ${PYTHON_BIN}"
    fi
}

setup_env() {
    check_system_dependencies

    log_info "Creating clean Python virtual environment (.venv)..."
    "${PYTHON_BIN}" -m venv .venv

    log_info "Activating virtual environment..."
    # shellcheck disable=SC1091
    source .venv/bin/activate

    log_info "Upgrading fundamental packaging tools (pip, setuptools, wheel)..."
    pip install --upgrade pip "setuptools<82" wheel

    log_info "Installing Subtone in editable mode (-e) along with development requirements..."
    pip install -e .
    pip install pytest tomli_w

    log_success "Subtone local development environment setup completed successfully!"
    echo -e "\nTo activate this environment in your current terminal session, run:"
    echo -e "  ${BOLD}source .venv/bin/activate${NC}\n"
}

run_tests() {
    if [[ ! -d ".venv" ]]; then
        log_error "Virtual environment not found! Please run './subtone.sh setup' first."
        exit 1
    fi

    log_info "Activating virtual environment and running tests..."
    # shellcheck disable=SC1091
    source .venv/bin/activate

    pytest
}

run_cli() {
    if [[ ! -d ".venv" ]]; then
        log_error "Virtual environment not found! Please run './subtone.sh setup' first."
        exit 1
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate

    log_info "Running: subtone $*"
    subtone "$@"
}

show_shell_instructions() {
    echo -e "${BOLD}To enter the Subtone virtual environment, run the following command in your terminal:${NC}"
    echo -e "\n  ${GREEN}source ${SCRIPT_DIR}/.venv/bin/activate${NC}\n"
    echo -e "Once activated, you can run the '${CYAN}subtone${NC}' CLI directly."
}

# Parse command line arguments
if [[ $# -lt 1 ]]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift

case "${COMMAND}" in
    setup)
        setup_env
        ;;
    test)
        run_tests
        ;;
    run)
        run_cli "$@"
        ;;
    shell)
        show_shell_instructions
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        log_error "Unknown command: ${COMMAND}"
        show_help
        exit 1
        ;;
esac
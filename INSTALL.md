# Installing Transcript as a UV Tool

## What is a UV Tool?

UV tools are standalone command-line applications that can be installed globally and run from anywhere on your system. Once installed, you can use `transcript` command directly without needing to run `python -m` or `uv run`.

## Installation Methods

### 1. Install from Local Directory (Development)

If you have cloned the repository:

```bash
cd /path/to/transcript
uv tool install .
```

### 2. Install from GitHub (Coming Soon)

Once published:

```bash
uv tool install git+https://github.com/yourusername/transcript.git
```

### 3. Install from PyPI (Future)

If published to PyPI:

```bash
uv tool install transcript
```

## Usage After Installation

Once installed as a uv tool, you can use it from anywhere:

```bash
# Show help
transcript --help

# Live transcription
transcript live --language pt --multilingual

# File transcription
transcript file --format srt --model large

# From any directory
cd ~/Documents
transcript file --input ./recordings --output ./transcripts
```

## Managing the Tool

### Check Installation

```bash
# List installed uv tools
uv tool list

# Check which transcript
which transcript
```

### Update

```bash
# From the project directory
cd /path/to/transcript
uv tool install . --force
```

### Uninstall

```bash
uv tool uninstall transcript
```

## Benefits of UV Tool Installation

1. **Global Access** - Run from any directory
2. **Isolated Environment** - Dependencies don't conflict with other projects
3. **Easy Updates** - Simple reinstall to update
4. **Clean System** - All dependencies contained in tool environment

## Troubleshooting

### Command Not Found

If `transcript` command is not found after installation:

1. Check if it's installed:

   ```bash
   uv tool list
   ```

2. Ensure `~/.local/bin` is in your PATH:

   ```bash
   echo $PATH
   ```

3. If not in PATH, add to your shell config:

   ```bash
   # For bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc

   # For zsh
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

### Import Errors

If you get module import errors, ensure you're using the installed version:

```bash
# Use the installed tool
transcript live

# Not the local development version
python -m transcript_pkg.cli live
```

## Development vs Production

- **Development**: Use `uv run python -m transcript_pkg.cli` or `uv run ./transcript`
- **Production**: Install as tool with `uv tool install .` and use `transcript`

The tool installation creates an isolated environment with all dependencies, making it perfect for end users who just want to use the transcription features without dealing with Python environments.

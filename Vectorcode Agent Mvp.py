"""
VectorCode Agent MVP for GitHub Repositories
AI-powered coding, documentation, and compliance-support agent.

What this version does:
- Works with a GitHub repository cloned locally
- Reads project files safely
- Reviews code and provides secure development guidance
- Supports documentation, testing, DevSecOps, and compliance workflows
- Can optionally create a working branch and commit generated documentation

Recommended GitHub workflow:
1. Install GitHub CLI: https://cli.github.com/
2. Authenticate: gh auth login
3. Clone your repository:
   gh repo clone vectorfortress/the-kofcity-project repos/the-kofcity-project
4. Point PROJECT_ROOT to that cloned repository.
5. Run the agent locally.
6. Review any changes manually.
7. Push a branch and open a pull request only after review.

Before running:
1. pip install openai python-dotenv
2. Create a .env file with:
   OPENAI_API_KEY=your_api_key_here
3. Optional:
   PROJECT_ROOT=./repos/the-kofcity-project
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Set this in .env or change the default path below.
GITHUB_OWNER = "vectorfortress"
GITHUB_REPO = "the-kofcity-project"
GITHUB_FULL_REPO = f"{GITHUB_OWNER}/{GITHUB_REPO}"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", f"./repos/{GITHUB_REPO}")).resolve()

# Basic safety controls.
BLOCKED_COMMANDS = [
    "rm -rf",
    "del /s",
    "format",
    "shutdown",
    "reboot",
    "curl | sh",
    "wget | sh",
    "git push --force",
    "git reset --hard",
]

ALLOWED_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".md",
    ".json", ".yml", ".yaml", ".txt", ".toml", ".env.example",
    ".rb", ".go", ".java", ".php", ".cs", ".rs"
}

IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    ".next", ".astro", "coverage", ".pytest_cache"
}


def ensure_project_root() -> None:
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Project root does not exist: {PROJECT_ROOT}\n"
            "Clone your GitHub repo first, then set PROJECT_ROOT in your .env file."
        )


def run_command(command: str, timeout: int = 60) -> str:
    """Run a command inside the project directory with basic safety checks."""
    ensure_project_root()
    lowered = command.lower()

    for blocked in BLOCKED_COMMANDS:
        if blocked in lowered:
            raise PermissionError(f"Blocked unsafe command: {command}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nEXIT CODE: {result.returncode}"


def list_files() -> List[str]:
    """List readable project files."""
    ensure_project_root()
    results = []

    for path in PROJECT_ROOT.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in ALLOWED_FILE_EXTENSIONS:
            results.append(str(path.relative_to(PROJECT_ROOT)))

    return sorted(results)


def read_file(relative_path: str) -> str:
    """Read a file from the project."""
    path = (PROJECT_ROOT / relative_path).resolve()

    if not str(path).startswith(str(PROJECT_ROOT)):
        raise PermissionError("Cannot read outside project root.")

    if path.suffix not in ALLOWED_FILE_EXTENSIONS:
        raise PermissionError(f"File type not allowed: {path.suffix}")

    return path.read_text(encoding="utf-8", errors="ignore")


def write_file(relative_path: str, content: str) -> str:
    """Write content to a project file."""
    path = (PROJECT_ROOT / relative_path).resolve()

    if not str(path).startswith(str(PROJECT_ROOT)):
        raise PermissionError("Cannot write outside project root.")

    if path.suffix not in ALLOWED_FILE_EXTENSIONS:
        raise PermissionError(f"File type not allowed: {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Updated {relative_path}"


def git_status() -> str:
    return run_command("git status --short")


def create_branch(branch_name: str) -> str:
    """Create a safe working branch for agent-generated work."""
    safe_name = branch_name.replace(" ", "-").lower()
    return run_command(f"git checkout -b {safe_name}")


def commit_changes(message: str) -> str:
    """Commit current changes locally. Review changes before using this."""
    status = git_status()
    if not status.strip().replace("STDOUT:", "").strip():
        return "No changes to commit."

    run_command("git add .")
    return run_command(f'git commit -m "{message}"')


def build_project_context(max_files: int = 18, max_chars_per_file: int = 3500) -> str:
    """Create a compact codebase summary for the agent."""
    files = list_files()[:max_files]
    sections = []

    for file in files:
        try:
            content = read_file(file)[:max_chars_per_file]
            sections.append(f"\n--- FILE: {file} ---\n{content}")
        except Exception as exc:
            sections.append(f"\n--- FILE: {file} ---\nCould not read file: {exc}")

    return "\n".join(sections)


def detect_project_type() -> str:
    """Infer the likely project type from common files."""
    files = set(list_files())

    if "package.json" in files:
        return "Node/JavaScript/TypeScript project"
    if "requirements.txt" in files or "pyproject.toml" in files:
        return "Python project"
    if "Gemfile" in files:
        return "Ruby project"
    if "go.mod" in files:
        return "Go project"
    if "pom.xml" in files or "build.gradle" in files:
        return "Java project"

    return "Unknown or static project"


def ask_agent(user_request: str) -> str:
    """Ask the coding agent for analysis and recommended changes."""
    project_context = build_project_context()
    project_type = detect_project_type()
    status = git_status()

    system_prompt = """
You are VectorCode Agent, a secure coding and compliance-support agent for GitHub repositories.

Your mission:
- Understand the user's development request.
- Review the codebase context provided.
- Recommend clean, secure, maintainable changes.
- Support documentation, testing, DevSecOps, and compliance workflows.
- When useful, provide GitHub issue, branch, commit, and pull request recommendations.

Important rules:
- Do not claim you changed files unless a tool actually changed them.
- Do not recommend deleting files unless clearly justified.
- Prioritize secure defaults.
- Prefer pull requests over direct commits to main.
- Never expose secrets, tokens, private keys, or credentials.
- When useful, map security concerns to practical control families such as access control, audit logging, configuration management, change management, vulnerability management, and system integrity.
- Provide clear next steps.
"""

    response = client.responses.create(
        model="gpt-5.5-thinking",
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
User request:
{user_request}

Repository path:
{PROJECT_ROOT}

Detected project type:
{project_type}

Git status:
{status}

Project context:
{project_context}
""",
            },
        ],
    )

    return response.output_text


def generate_repo_review() -> str:
    """Generate a KofCity-focused repository review and save it as documentation."""
    review = ask_agent(
        "Review the KofCity project repository for code quality, security, maintainability, testing gaps, documentation gaps, deployment readiness, payment workflow risks, role-based access needs for parents, schools, vendors, riders, and admins, and compliance considerations."
    )

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    report = f"# VectorCode Repository Review\n\nGenerated: {timestamp}\n\n{review}\n"
    write_file("docs/vectorcode-repo-review.md", report)
    return "Created docs/vectorcode-repo-review.md"


def main() -> None:
    print("VectorCode Agent MVP for GitHub")
    print("GitHub repo:", GITHUB_FULL_REPO)
    print("Repository:", PROJECT_ROOT)
    print("Type 'exit' to quit.")
    print("Useful commands:")
    print("- review repo")
    print("- git status")
    print("- create branch vectorcode/kofcity-review")
    print("- commit changes")
    print("- create kofcity issue")
    print("- create pr")
    print("- Or ask any coding/compliance question.
")

    while True:
        user_request = input("What should the agent do? > ").strip()

        if user_request.lower() in {"exit", "quit"}:
            break

        try:
            if user_request.lower() == "git status":
                print(git_status())
            elif user_request.lower() == "review repo":
                print(generate_repo_review())
            elif user_request.lower().startswith("create branch "):
                branch = user_request.replace("create branch ", "", 1).strip()
                print(create_branch(branch))
            elif user_request.lower() == "commit changes":
                print(commit_changes("Add VectorCode Agent review for KofCity project"))
            elif user_request.lower() == "create kofcity issue":
                issue_title = "VectorCode review: improve KofCity security, testing, and deployment readiness"
                issue_body = """## Summary
Use VectorCode Agent to review and improve the KofCity project codebase.

## Focus Areas
- Parent/customer food ordering workflow
- School delivery coordination workflow
- Vendor order management workflow
- Rider delivery workflow
- Admin management workflow
- Payment methods: MTN MoMo, card, and cash
- Security and role-based access control
- Testing and deployment readiness
- Documentation and operational support

## Acceptance Criteria
- Repository review is generated under `docs/vectorcode-repo-review.md`
- Security and maintainability gaps are documented
- Recommended improvements are converted into actionable tasks
- Pull request is opened for review before merging
"""
                safe_body = issue_body.replace('"', '\"')
                print(run_command(f'gh issue create --repo {GITHUB_FULL_REPO} --title "{issue_title}" --body "{safe_body}"'))
            elif user_request.lower() == "create pr":
                print(run_command('git push -u origin HEAD'))
                print(run_command('gh pr create --fill'))
            else:
                answer = ask_agent(user_request)
                print("\n--- Agent Response ---\n")
                print(answer)
                print("\n----------------------\n")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()

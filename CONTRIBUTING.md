# Contributors

The project differentiates between 3 levels of contributors:

- Contributors: people who have contributed before (no special privileges)
- Collaborators (Triage): people with significant contributions, who may be responsible for some parts of the code, and are expected to maintain and review contributions for the code they own
- Maintainers: responsible for reviewing and merging PRs, after approval from the code owners

# AI Usage Policy

AI-assisted development is allowed in this repository.

Contributors using AI must still:

1. Review generated changes carefully.
2. Understand the code and design they submit.
3. Be able to explain architectural choices and behavior changes.
4. Validate changes with appropriate tests, benchmarks, or manual checks.

For repository-specific expectations around AI-agent behavior, see [AGENTS.md](AGENTS.md).

# Pull requests (for contributors & collaborators)

Before submitting your PR:
- Search for existing PRs to prevent duplicating efforts
- Test your changes:
    - Execute the relevant local test coverage before publishing
    - Verify that perplexity and performance are not affected negatively when you change inference behavior
    - If you modify `ggml` operators or backend-sensitive code, run the relevant backend consistency tests
- Create separate PRs for each feature or fix:
    - Avoid combining unrelated changes in a single PR
    - For intricate features, consider opening a feature request first to discuss and align expectations
    - When adding support for a new model or feature, focus on **CPU support only** in the initial PR unless you have a good reason not to. Add support for other backends like CUDA in follow-up PRs
- Consider allowing write access to your branch for faster reviews, as reviewers can push commits directly
- If you are a new contributor, limit your open PRs to 1.

After submitting your PR:
- Expect requests for modifications to ensure the code meets the project's standards for quality and long-term maintainability
- Maintainers will rely on your insights and approval when making a final decision to approve and merge a PR
- If your PR becomes stale, rebase it on top of latest `master` to get maintainers attention
- Consider adding yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for fixing related issues and reviewing related PRs

# Pull requests (for maintainers)

- Squash-merge PRs
- Use the following format for the squashed commit title: `<module> : <commit title> (#<issue_number>)`
- Let other maintainers merge their own PRs
- When merging a PR, make sure you have a good understanding of the changes
- Be mindful of maintenance: most of the work going into a feature happens after the PR is merged. If the PR author is not committed to contribute long-term, someone else needs to take responsibility

# Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types such as `int32_t` in the public API, e.g. `size_t` may also be appropriate for allocation sizes or byte offsets
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`
    - In C++ code omit optional `struct` and `enum` keyword whenever they are not necessary
- Try to follow the existing patterns in the code (indentation, spaces, etc.). In case of doubt use `clang-format` (from clang-tools v15+) to format the added code
- For anything not covered in the current guidelines, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices

# Naming guidelines

- Use `snake_case` for function, variable and type names
- Enum values are always in upper case and prefixed with the enum name
- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>`
- C/C++ filenames are all lowercase with dashes. Headers use the `.h` extension. Source files use the `.c` or `.cpp` extension
- Python filenames are all lowercase with underscores

# Code maintenance

- Existing code should have designated collaborators and/or maintainers specified in the [CODEOWNERS](CODEOWNERS) file responsible for:
  - Reviewing and merging related PRs
  - Fixing related bugs
  - Providing developer guidance/support

- When adding or modifying a large piece of code:
  - If you are a collaborator, make sure to add yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for reviewing related PRs
  - If you are a contributor, find an existing collaborator who is willing to review and maintain your code long-term
  - Provide the necessary CI workflow (and hardware) to test your changes

# Documentation

- Documentation is a community effort
- When you need to look into the source code to figure out how to use an API consider adding a short summary to the header file for future reference
- When you notice incorrect or outdated documentation, please update it

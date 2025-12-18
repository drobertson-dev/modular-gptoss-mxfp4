# Testing Mojo Latest Conventions

From the internal docs, Mojo’s current testing story is:

The old mojo test CLI command was removed on October 31, 2025.

The recommended way now is to use the standard-library testing framework: assertion helpers in testing plus the TestSuite struct as your “runner,” and execute tests with mojo run. fileciteturn0file4turn0file11

So there is a built‑in runner, but it’s a library type (TestSuite), not a separate CLI command.

How to write tests in Mojo
A “unit test” in Mojo is just a function that meets these rules:

Name starts with test_ (for automatic discovery).

Takes no arguments.

Returns None.

Raises an error to indicate failure.

Is defined at module scope, not as a struct method.

You normally use the assertion utilities from the testing module:

assert_true, assert_false

assert_equal, assert_not_equal, assert_almost_equal

assert_raises (context manager) fileciteturn0file2turn0file15

Example:

from testing import assert_equal, assert_true, TestSuite

def add(a: Int, b: Int) -> Int:
    return a + b

def test_add_basic():
    assert_equal(add(1, 2), 3)

def test_add_properties():
    assert_true(add(0, 5) == 5)
    assert_true(add(3, 4) == 7)

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
Key points here:

Any function whose name starts with test_and matches the signature rules will be picked up automatically by discover_tests. fileciteturn0file10turn0file5

__functions_in_module() is an intrinsic that returns all functions in the current module; TestSuite.discover_tests[...]() filters that list to test functions and builds a suite. fileciteturn0file5turn0file19

The built‑in runner: TestSuite
TestSuite is the built‑in test runner type in the standard library. It supports:

Automatic discovery via TestSuite.discover_tests[__functions_in_module()]()

Manual registration via suite.test[some_test]()

Running and printing a colored, summarized report with .run() fileciteturn0file11turn0file8

Minimal discovery-based usage:

from testing import assert_equal, TestSuite

def test_something():
    assert_equal(1 + 1, 2)

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
Manual registration variant:

from testing import assert_equal, TestSuite

def some_test():
    assert_equal(2 + 2, 4)

def main():
    var suite = TestSuite()
    suite.test[some_test]()     # register the test
    suite^.run()                # run and print report
TestSuite.run() executes all registered tests, prints per‑test PASS/FAIL lines and a final summary, and exits with a non‑zero status if any tests fail. fileciteturn0file1turn0file8

Running tests from the CLI
Because mojo test is gone, you run test files like any other Mojo program:

mojo run test_my_module.mojo
If your main() uses TestSuite.discover_tests[__functions_in_module()]().run(), Mojo will:

Discover all test_ functions in that file.

Run them.

Print a summary and exit non‑zero if there were failures. fileciteturn0file4turn0file1

TestSuite also understands some CLI flags passed after the file name:

--skip test_name1 test_name2 – skip specific tests.

--only test_name1 test_name2 – run only those tests (others are skipped).

--skip-all – collect but skip all tests. fileciteturn0file9turn0file8

Example:

mojo run test_my_module.mojo --only test_add_basic
Extras: property-based testing (optional)
If you want QuickCheck/Hypothesis-style property tests, there’s also a testing.prop package with:

PropTest and PropTestConfig as a property test runner and configuration.

Strategy and strategy.* for generating random input values. fileciteturn0file6turn0file18

But for most unit tests, plain TestSuite + assertions is the “best practice” path.

Summary
Best way today: Write test_... functions using the testing module’s assertions, then use TestSuite.discover_tests[__functions_in_module()]().run() in main(), and run with mojo run your_test_file.mojo. fileciteturn0file4turn0file11

Built‑in runner? Yes: TestSuite in the standard library is the built‑in test runner; the old mojo test command is deprecated and removed.

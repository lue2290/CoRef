"""Run focused numerical tests with the preserved research import roots."""
from pathlib import Path
import runpy,unittest,sys,argparse
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'run.py'),run_name='_vrdd_test_environment')
suite=unittest.TestSuite()
for pattern in ['test_reference_net.py','test_reference_f1.py','test_release.py']:
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-archive-tests',action='store_true',help='Require original dataset annotations and retired comparison-head source')
    args=parser.parse_args()
    if not args.include_archive_tests:
        def cases(group):
            for test in group:
                if isinstance(test,unittest.TestSuite):yield from cases(test)
                else:yield test
        integration={'test_all30_parents_full_bank_corrected_game_and_task2','test_decoding_matches_completed156_f0_and_f1'}
        for test in cases(suite):
            if test._testMethodName in integration:
                method=getattr(type(test),test._testMethodName)
                method.__unittest_skip__=True
                method.__unittest_skip_why__='Requires full historical dataset/comparison-head workspace; use --include-archive-tests there'
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

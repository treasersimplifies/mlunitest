import unittest
import deps.HTMLTestRunner as HTMLTestRunner

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        print("Start testing...")

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_upper2(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_upper3(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def tearDown(self):
        print("End testing...")

if __name__ == "__main__":
    # 测试用例保存的目录
    case_dirs = "/Users/treasersmac/Programming/LogicRegression-UnitTest"
    # 加载测试用例
    discover = unittest.defaultTestLoader.discover(case_dirs, "*_test.py")
    # 运行测试用例同时保存测试报告
    test_report_path = "/Users/treasersmac/Programming/LogicRegression-UnitTest/reports/report.html"
    with open(test_report_path, "wb+") as report_file:
        runner = HTMLTestRunner.HTMLTestRunner(stream=report_file, title="软件单元测试报告", description="机器学习（逻辑回归）")
        runner.run(discover)


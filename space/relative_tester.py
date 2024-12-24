from two_sample_tester import two_sample_tester


class RelativeTester:
    def __init__(self):
        print("Relative Tester init")

    def test(self, input_text):
        print("Relative Tester test")
        two_sample_tester.test(input_text)
        two_sample_tester.test(input_text)
        return f"Relative Tester result {input_text}"


relative_tester = RelativeTester()

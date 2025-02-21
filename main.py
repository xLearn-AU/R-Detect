from relative_tester import relative_tester
from utils import init_random_seeds

if __name__ == "__main__":
    init_random_seeds()
    with open("demo_text_gpt.txt", "r") as file:
        content = file.read()
        print(relative_tester.test(content))
    # print(content)

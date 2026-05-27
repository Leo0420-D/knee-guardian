import pkg_resources

def generate_requirements(output_file='requirements.txt'):
    try:
        with open(output_file, 'w') as f:
            for dist in pkg_resources.working_set:
                req = dist.as_requirement()
                f.write(f"{req}\n")
        print(f"成功生成{output_file}")
    except Exception as e:
        print(f"生成requirements.txt时出错: {e}")

if __name__ == "__main__":
    generate_requirements()
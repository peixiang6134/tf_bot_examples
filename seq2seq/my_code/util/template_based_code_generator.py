class TemplateBasedCodeGenerator:

    @staticmethod
    def generate(lines, config):
        """"generate content by extracting proper parts from the template file based on the given configuration
            :param lines: list[str], list of lines from the template file name. The lines contains variables to be
                initiated and if/elif/endif flow controls
            :param config: dict, configuration map
        """
        filtered_content = []
        TemplateBasedCodeGenerator._filter_lines(lines, 0, config, filtered_content)
        return filtered_content

    @staticmethod
    def _evaluate(s, config):
        """evaluate expression s after initiating variables from config"""
        for k, v in config.items():
            s = str.replace(s, "$" + k + "$", str(v))
        try:
            return eval(s)
        except Exception as e:
            print(str(e) + ": " + s)
            raise Exception(e)

    @staticmethod
    def _filter_lines(lines, start, config, filtered_content):
        """
        filter contents from lines given starting position and configuration
        :param lines: list[str], input lines
        :param start: int, starting position
        :param config: dict, containing configuration variables and their values
        :param filtered_content: list[str], output
        :return:
        """
        index = start
        meet_if = False
        while index < len(lines):
            line = lines[index].rstrip()
            if str.startswith(line, "#if "):
                meet_if = True
                if TemplateBasedCodeGenerator._evaluate(line[len("#if "):], config):
                    """run the if branch """
                    index = TemplateBasedCodeGenerator._filter_lines(lines, index + 1, config, filtered_content)
                    index = TemplateBasedCodeGenerator._advance_until(lines, index, ["#endif"])
                else:
                    """check other branches or endif"""
                    index = TemplateBasedCodeGenerator._advance_until(lines, index + 1, ["#elif ", "#else", "#endif"])
            elif str.startswith(line, "#elif "):
                if not meet_if:
                    """it means we are in the if branch, and should stop here"""
                    return index
                if TemplateBasedCodeGenerator._evaluate(line[len("#elif "):], config):
                    """run the elif branch """
                    index = TemplateBasedCodeGenerator._filter_lines(lines, index + 1, config, filtered_content)
                    index = TemplateBasedCodeGenerator._advance_until(lines, index, ["#endif"])
                else:
                    """check other branches or endif"""
                    index = TemplateBasedCodeGenerator._advance_until(lines, index + 1, ["#elif ", "#else", "#endif"])
            elif str.startswith(line, "#else"):
                if not meet_if:
                    """it means we are in the if branch, and should stop here"""
                    return index
                    """run the else branch """
                index = TemplateBasedCodeGenerator._filter_lines(lines, index + 1, config, filtered_content)
                index = TemplateBasedCodeGenerator._advance_until(lines, index, ["#endif"])
            elif str.startswith(line, "#endif"):
                if not meet_if:
                    """it means we are in the if or elif branch, and should stop here"""
                    return index
                """it means we didn't enter any if or elif branches, we skip this endif, and move on."""
                index += 1
                meet_if = False
            else:
                for k, v in config.items():
                    line = line.replace("$" + k + "$", str(v))
                filtered_content.append(line)
                index += 1
        return len(lines)

    @staticmethod
    def _advance_until(lines, index, tags):
        """keep skipping lines until seeing lines starting with tags"""

        i = index
        while True:
            if i < len(lines):
                line = lines[i]
                if any([str.startswith(line, tag) for tag in tags]):
                    return i
                if str.startswith(line, "#if "):
                    """need a recursive call to skip the inner if/endif block"""
                    i = TemplateBasedCodeGenerator._advance_until(line, i + 1, ["#endif"])
                i += 1
        assert False

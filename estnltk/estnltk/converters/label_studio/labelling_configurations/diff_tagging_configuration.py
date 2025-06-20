import random

from typing import Literal

from estnltk.converters.label_studio.labelling_configurations.default_color_schemes import (
    DEFAULT_BG_COLORS,
)


class DiffTaggingConfiguration:
    """
    A simple way to define configuration file for Label Studio diff tagging tasks.
    In such a task the goal is to pick the correct annotation between conflicting
    layer annotations. Fields of the configuration object specify all available
    configuration options. The result can be accessed through str() function.
    """

    def __init__(
        self,
        layer_a: str,
        layer_b: str,
        class_labels: list[str] | dict[str, str],
        granularity: Literal["word", "symbol"] = "word",
        header: str | None = None,
        rand_seed: int | None = None,
    ):
        """
        Defines class labels that are used in the labelling task together with background colors of selections.
        If the parameter 'class_labels' is a list then default coloring scheme is used.
        To specify background colors, the parameter 'class_labels' must be a dictionary of label-color pairs.
        By default, the labeling window does not have a header and selection granularity is word level.

        It is possible to set hotkeys (keyboard shortcuts) for classes.
        For that one has to use appropriately named properties.

        Finally, it is possible to specify the name of the text field in the Labels Studio data to be displayed
        and labelled by setting the field input_text_field. The value must match the field name that contains the
        text to be labelled in your json exports. By default, this is set to be 'text_a'.
        """
        self.set_class_labels(class_labels, rand_seed=rand_seed)

        self.layer_a = layer_a
        self.layer_b = layer_b

        self.text_element_a = "text_a"
        self.text_element_b = "text_b"

        self.input_text_field_a = "text_a"
        self.input_text_field_b = "text_b"

        self.choices = [
            {"value": "A", "alias": "text_a"},
            {"value": "B", "alias": "text_b"},
            {"value": "None"},
        ]

        self.header = header
        self.annotator_element = "validated_annotation"

        if granularity in ["symbol", "word"]:
            self.granularity = granularity
        else:
            raise ValueError("Parameter granularity can have values 'symbol' or 'word'")

    def set_class_labels(
        self,
        class_labels: list[str] | dict[str, str],
        rand_seed: int | None = None,
    ):
        """
        Defines class labels that are used in the labelling task together with background colors of selections.
        For existing configurations, this process clears other class-specific settings such as hotkeys and aliases.
        """
        if isinstance(class_labels, list):
            colors = DEFAULT_BG_COLORS.get(len(class_labels), None)
            if colors is None:
                if rand_seed is not None:
                    random.seed(rand_seed)

                colors = [
                    f"#{random.randint(0, 0xFFFFFF):06X}"
                    for _ in range(len(class_labels))
                ]

            self.class_labels = [
                {"value": label, "background": colors[i]}
                for i, label in enumerate(class_labels)
            ]

        elif isinstance(class_labels, dict):
            self.class_labels = [
                {"value": label, "background": color}
                for label, color in class_labels.items()
            ]

        else:
            raise ValueError(
                "Expecting to see class_labels as a list of labels or dictionary of label-color pairs"
            )

    @property
    def background_colors(self):
        return {
            item["value"]: item.get("background", None) for item in self.class_labels
        }

    @background_colors.setter
    def background_colors(self, values: dict[str, str]):
        if not isinstance(values, dict):
            raise ValueError(
                "Expecting to see input as a dictionary of label-color pairs"
            )
        if len(values) != len(self.class_labels):
            raise ValueError(
                "The number of label-color pairs does not match with class count"
            )

        index = {item["value"]: loc for loc, item in enumerate(self.class_labels)}
        for label, color in values.items():
            loc = index.get(label, None)
            if loc is None:
                raise ValueError("Unknown class label inside a label-color pair")
            self.class_labels[loc]["background"] = color

    @property
    def hotkeys(self):
        return {item["value"]: item.get("hotkey", None) for item in self.choices}

    @hotkeys.setter
    def hotkeys(self, values: dict[str, str]):
        if not isinstance(values, dict):
            raise ValueError(
                "Expecting to see input as a dictionary of label-hotkey pairs"
            )
        if len(values) != len(self.choices):
            raise ValueError(
                "The number of label-hotkey pairs does not match with class count"
            )

        index = {item["value"]: loc for loc, item in enumerate(self.choices)}
        for label, hotkey in values.items():
            loc = index.get(label, None)
            if loc is None:
                raise ValueError("Unknown class label inside a label-hotkey pair")
            self.choices[loc]["hotkey"] = hotkey

    def __str__(self):
        """
        Outputs the XML labeling interface file for Label Studio
        """
        result = "<View>\n"
        if self.header is not None:
            result += f'  <Header value="{self.header}" />\n'

        result += f'  <Choices name="{self.annotator_element}" toName="{self.input_text_field_a}" choice="single-radio" showInline="true" >\n    '
        choice_tags = []
        for choice_dict in self.choices:
            attributes = " ".join(f'{k}="{v}"' for k, v in choice_dict.items())
            choice_tags.append(f"<Choice {attributes} />")
        result += "\n    ".join(choice_tags) + "  </Choices>\n"

        result += f'  <Labels name="predicted_labels" toName="{self.input_text_field_a}" visible="false" >\n    '
        label_tags = []
        for label_dict in self.class_labels:
            attributes = " ".join(f'{k}="{v}"' for k, v in label_dict.items())
            label_tags.append(f"<Label {attributes} />")
        result += "\n    ".join(label_tags) + "\n  </Labels>"

        result += f'\n  <Header value="Annotation A" />'
        result += f'\n  <Text name="{self.text_element_a}" value="${self.input_text_field_a}" granularity="{self.granularity}" showLabels="true" />'
        result += f'\n  <Header value="Annotation B" />'
        result += f'\n  <Text name="{self.text_element_b}" value="${self.input_text_field_b}" granularity="{self.granularity}" showLabels="true" />'
        result += "\n</View>"

        return result

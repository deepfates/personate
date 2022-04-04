from personate.utils.logger import logger

def get_annotation_from_template(data):
    name = data.get("name", None)
    preset = data.get("preset", None)
    template_path = "personate.meta.templates.{}".format(preset)
    logger.debug(f"Using template {template_path}")
    if preset == "custom":
        template = __import__(template_path, fromlist=["template"]).template

        introduction = data.get("introduction", None)

        if not introduction:
            raise ValueError("You need to specify an introduction for your agent.")

        template["introduction"] = introduction
    else:
        template = __import__(template_path, fromlist=["template"]).template

        introduction = data.get("introduction", None)
        if not introduction and "{introduction}" in template["introduction"]:
            raise ValueError(
                "You need to introduce your agent. Have a look at the example template."
            )
        if introduction:
            template["introduction"] = template["introduction"].replace(
                "{introduction}", introduction
            )

        logger.debug(f"Using introduction {template['introduction']}")

    template = eval(str(template).replace("{chatbot_name}", name))
    return template

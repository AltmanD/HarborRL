"""Offline configuration discovery and generation; no optional runtime imports."""


def build_catalog():
    from .catalog import build_catalog as build
    return build()


def build_config_bundle(answers, *, output_directory):
    from .wizard import build_config_bundle as build
    return build(answers, output_directory=output_directory)

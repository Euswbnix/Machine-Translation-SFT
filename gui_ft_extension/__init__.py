"""Fine-tuned model registry hooked into the desktop GUI of `Machine_translation`.

When the user toggles 'Show fine-tuned models' in the Advanced menu, the
GUI tries to import `gui_ft_extension.ft_models`. If this package is on
the PYTHONPATH (i.e., this repo lives next to Machine_translation/), the
extra entries declared here appear in the dropdown.
"""

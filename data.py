import yaml

def get_atlas_data(atlas_data_filepath: str):
    """_summary_

    Args:
        atlas_data_filepath (str): _description_

    Returns:
        _type_: _description_
    """
    with open(atlas_data_filepath) as f:
        # Parse YAML
        data = yaml.safe_load(f)

        first_matrix = data['matrices'][0]
        tactics = first_matrix['tactics']
        techniques = first_matrix['techniques']

        studies = data['case-studies']
        return tactics, techniques, studies
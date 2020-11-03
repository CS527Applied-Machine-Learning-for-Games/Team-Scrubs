def _load_properties(filename):
    properties = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split("=")
            if len(parts) < 2:
                continue
            properties[parts[0].strip()] = int(parts[1].strip())
    return properties


def _write_properties(filename, param, value):
    # Read file and find the correct value to replace
    new_properties = ""
    found = False
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split("=")
            if len(parts) == 2 and parts[0]==param:
                new_properties += param + "=" + str(value) + "\n" # Note the type of value is str
                found = True
            else:
                new_properties += line
    if not found:
        new_properties += "\n" + param + "=" + str(value)
    # Overwrite properties file with new properties
    with open(filename, "w") as f:
        f.write(new_properties)


def load_config():
    return _load_properties("config.properties")


def load_state():
    return _load_properties("state.properties")


def write_config(param, value):
    _write_properties("config.properties", param, value)


def write_state(param, value):
    _write_properties("state.properties", param, value)
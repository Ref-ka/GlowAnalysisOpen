import json


def get_coordinates(labels_file):
    with open(labels_file) as file:
        data = json.load(file)

    coordinates = []
    for image in data:
        if image["drafts"]:
            coordinates.append(
                [
                    round((image['drafts'][0]["result"][0]["value"]["x"] / 100) * 256),
                    round((image['drafts'][0]["result"][0]["value"]["y"] / 100) * 256),
                    round(((image['drafts'][0]["result"][0]["value"]["x"] + image['drafts'][0]["result"][0]["value"][
                        "width"]) / 100) * 256),
                    round(((image['drafts'][0]["result"][0]["value"]["y"]) + image['drafts'][0]["result"][0]["value"][
                        "height"]) / 100 * 256)
                ]
            )
        else:
            coordinates.append(
                [
                    round((image['annotations'][0]["result"][0]["value"]["x"] / 100) * 256),
                    round((image['annotations'][0]["result"][0]["value"]["y"] / 100) * 256),
                    round(
                        ((image['annotations'][0]["result"][0]["value"]["x"] + image['annotations'][0]["result"][0]["value"][
                            "width"]) / 100) * 256),
                    round(((image['annotations'][0]["result"][0]["value"]["y"]) +
                           image['annotations'][0]["result"][0]["value"][
                               "height"]) / 100 * 256)
                ]
            )
    return coordinates


if __name__ == "__main__":
    print(get_coordinates("project-2-at-2025-03-28-21-19-2a62a146.json"))

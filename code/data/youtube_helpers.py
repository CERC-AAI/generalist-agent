# constants used for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0"
BASE = 'https://www.youtube.com'
ENDPOINT = '/youtubei/v1/next?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8&prettyPrint=false'
PAYLOAD = {
    "context": {
        "client": {
            "clientName": "WEB",
            "clientVersion": "2.20230824.00.00",
            "newVisitorCookie": True,
        },
        "user": {
            "lockedSafetyMode": False,
        }
    }
}


def getValue(source, path):
    value = source
    for key in path:
        if type(key) is str:
            if key in value.keys():
                value = value[key]
            else:
                value = None
                break
        elif type(key) is int:
            if len(value) != 0:
                value = value[key]
            else:
                value = None
                break
    return value


def parse_response(response):
    videorenderers = getValue(response, ["playerOverlays", "playerOverlayRenderer", "endScreen", "watchNextEndScreenRenderer", "results"])
    videos = []

    if videorenderers is None:
        return videos

    for video in videorenderers:
        if "endScreenVideoRenderer" in video.keys():
            video = video["endScreenVideoRenderer"]
            j = {
                "isPlaylist" : False,
                "id": getValue(video, ["videoId"]),
                "thumbnails": getValue(video, ["thumbnail", "thumbnails"]),
                "title": getValue(video, ["title", "simpleText"]),
                "channel": {
                    "name": getValue(video, ["shortBylineText", "runs", 0, "text"]),
                    "id": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "browseId"]),
                    "link": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "canonicalBaseUrl"]),
                },
                "duration": getValue(video, ["lengthText", "simpleText"]),
                "accessibility": {
                    "title": getValue(video, ["title", "accessibility", "accessibilityData", "label"]),
                    "duration": getValue(video, ["lengthText", "accessibility", "accessibilityData", "label"]),
                },
                "link": "https://www.youtube.com" + getValue(video, ["navigationEndpoint", "commandMetadata", "webCommandMetadata", "url"]),
                "isPlayable": getValue(video, ["isPlayable"]),
                "videoCount": 1,
            }
            videos.append(j)

        if "endScreenPlaylistRenderer" in video.keys():
            video = video["endScreenPlaylistRenderer"]
            j = {
                "isPlaylist" : True,
                "id": getValue(video, ["playlistId"]),
                "thumbnails": getValue(video, ["thumbnail", "thumbnails"]),
                "title": getValue(video, ["title", "simpleText"]),
                "channel": {
                    "name": getValue(video, ["shortBylineText", "runs", 0, "text"]),
                    "id": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "browseId"]),
                    "link": getValue(video, ["shortBylineText", "runs", 0, "navigationEndpoint", "browseEndpoint", "canonicalBaseUrl"]),
                },
                "duration": getValue(video, ["lengthText", "simpleText"]),
                "accessibility": {
                    "title": getValue(video, ["title", "accessibility", "accessibilityData", "label"]),
                    "duration": getValue(video, ["lengthText", "accessibility", "accessibilityData", "label"]),
                },
                "link": "https://www.youtube.com" + getValue(video, ["navigationEndpoint", "commandMetadata", "webCommandMetadata", "url"]),
                "isPlayable": getValue(video, ["isPlayable"]),
                "videoCount": getValue(video, ["videoCount"]),
            }
            videos.append(j)
    return videos
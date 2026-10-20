from hikerapi import Client

cl = Client(token="9hqfcavw6vkiu2sprd8s1fmdh69w5c3n")

# Get user profile
user = cl.user_by_username_v1("narendramodi")
print(user["follower_count"])  # 612000000

# Get user's recent media
medias, end_cursor = cl.user_medias_chunk_v1(user_id=user["pk"])
for media in medias[:1]:
    print(media["code"], media["like_count"], media["pk"])

    comments = cl.media_comments_v2(id=media["pk"])

    print(comments["response"]["comment_count"], len(comments["response"]["comments"]), comments["response"]["caption"]["text"])

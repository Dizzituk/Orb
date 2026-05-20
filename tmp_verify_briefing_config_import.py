from app.briefing.briefing_config import get_topics


def main() -> None:
    topics = get_topics("command_centre_news")
    print("ok", len(topics))


if __name__ == "__main__":
    main()

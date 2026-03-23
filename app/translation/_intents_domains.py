# FILE: app/translation/_intents_domains.py
"""
Domain intent definitions for ASTRA's unified cross-domain routing.

These intents route user messages to the correct domain service
(finance, investments, content, social, lifestyle, debug, education,
builds) so the LLM can read real data and take actions across the
entire ASTRA system.

v3.0 (2026-03-13): Initial — 8 domain intents for unified routing.
"""
from __future__ import annotations

from typing import Dict

from .schemas import CanonicalIntent, IntentDefinition

DOMAIN_INTENTS: Dict[CanonicalIntent, IntentDefinition] = {

    # =========================================================================
    # FINANCE
    # =========================================================================
    CanonicalIntent.DOMAIN_FINANCE: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_FINANCE,
        description="Finance queries: budgets, expenses, tax, bank accounts, spending",
        behavior="Pull finance summary from finance service, inject into LLM context",
        trigger_phrases=[
            "how much did I spend",
            "what's my budget",
            "show my expenses",
            "bank balance",
            "monthly spending",
            "tax return",
            "credit card",
            "my finances",
            "financial summary",
            "how much have I earned",
            "income this month",
            "outgoings",
            "go to accounts",
            "open accounts",
            "show accounts",
            "move to accounts",
            "switch to accounts",
            "accounts tab",
            "go to finance",
            "open finance",
        ],
        trigger_patterns=[
            r"\b(budget|spend|spent|expense|earning|income|tax|outgoing|bank|financ|accounting|deliver|mileage|van cost|petrol|fuel)\b",
            r"\b(credit card|debit|salary|invoice|receipt)\b",
            r"\bhow much (did|have|do) (I|we) (spend|earn|owe|save)\b",
            r"\b(monthly|weekly|annual) (spend|cost|budget|income)\b",
        ],
    ),

    # =========================================================================
    # INVESTMENTS
    # =========================================================================
    CanonicalIntent.DOMAIN_INVESTMENTS: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_INVESTMENTS,
        description="Investment queries: portfolio, positions, stocks, crypto, performance",
        behavior="Pull portfolio data from investments service, inject into LLM context",
        trigger_phrases=[
            "how's my portfolio",
            "my investments",
            "stock price",
            "crypto portfolio",
            "trading 212",
            "best performer",
            "worst performer",
            "portfolio value",
            "dividend",
            "my positions",
            "market update",
            "how are my stocks",
            "investment summary",
            "go to investments",
            "open investments",
            "show investments",
            "move to investments",
            "switch to investments",
            "investment tab",
        ],
        trigger_patterns=[
            r"\b(portfolio|stock|share|crypto|bitcoin|ethereum|invest|trading|position|dividend)\b",
            r"\b(bull|bear|market|nasdaq|s&p|ftse)\b",
            r"\bhow.*(portfolio|investment|stock|share|position).*(doing|performing|going)\b",
            r"\b(buy|sell|hold)\s+(stock|share|crypto|position)\b",
        ],
    ),

    # =========================================================================
    # CONTENT
    # =========================================================================
    CanonicalIntent.DOMAIN_CONTENT: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_CONTENT,
        description="Content pipeline: creation, scheduling, publishing, review queue",
        behavior="Pull content state from content service, inject into LLM context",
        trigger_phrases=[
            "schedule my next short",
            "content in review",
            "publish the video",
            "what content is ready",
            "content pipeline",
            "my youtube channel",
            "create a short",
            "content output",
            "video production",
            "what's in the content queue",
            "upload to youtube",
            "content status",
            "go to content",
            "open content",
            "show content",
            "move to content",
            "switch to content",
            "content tab",
        ],
        trigger_patterns=[
            r"\b(content|video|short|youtube|publish|schedule|upload|thumbnail|caption)\b",
            r"\b(content\s*(pipeline|queue|status|output|review))\b",
            r"\b(create|make|produce|edit|cut)\s+(a\s+)?(short|video|reel|carousel|post)\b",
            r"\b(queue|schedule)\s+(for|to|on)\s+(youtube|tiktok|instagram|facebook)\b",
        ],
    ),

    # =========================================================================
    # SOCIAL MEDIA
    # =========================================================================
    CanonicalIntent.DOMAIN_SOCIAL: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_SOCIAL,
        description="Social media: comments, engagement, posting, analytics across platforms",
        behavior="Pull social media state from engagement service, inject into LLM context",
        trigger_phrases=[
            "check my comments",
            "reply to comments",
            "social media analytics",
            "engagement rate",
            "how are my posts doing",
            "post to tiktok",
            "post to instagram",
            "post to facebook",
            "social media status",
            "followers",
            "new comments",
            "go to social media",
            "open social media",
            "show social media",
            "move to social media",
            "switch to social media",
            "social media tab",
        ],
        trigger_patterns=[
            r"\b(comment|engage|follower|repost|reply|mention|dm)\b",
            r"\b(like|share)s?\s+(count|rate|ratio)\b",
            r"\b(social\s*media|tiktok|instagram|facebook|twitter)\s*(analytic|status|update|post)\b",
            r"\b(engagement|reach|impression)\s*(rate|metric|stat)\b",
        ],
    ),

    # =========================================================================
    # LIFESTYLE
    # =========================================================================
    CanonicalIntent.DOMAIN_LIFESTYLE: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_LIFESTYLE,
        description="Lifestyle: fitness, diet, keto, bodyboarding, wellness goals",
        behavior="Pull lifestyle data from lifestyle service, inject into LLM context",
        trigger_phrases=[
            "my body fat",
            "log my workout",
            "keto status",
            "what should I eat",
            "fitness goals",
            "track my weight",
            "bodyboarding",
            "surf conditions",
            "my diet",
            "calorie count",
            "macro breakdown",
            "exercise routine",
            "sugar intake",
            "health goals",
            "how am I doing with fitness",
            "go to health",
            "open health",
            "show health",
            "move to health",
            "switch to health",
            "health tab",
            "go to fitness",
            "open fitness",
        ],
        trigger_patterns=[
            r"\b(workout|exercise|gym|fitness|body\s*fat|weight|keto|diet|calorie|macro|surf\w*|bodyboard\w*)\b",
            r"\b(log|track|record)\s+(my\s+)?(workout|exercise|weight|meal|food|run)\b",
            r"\b(how.*(fitness|health|diet|weight|keto).*going)\b",
        ],
    ),

    # =========================================================================
    # DEBUG
    # =========================================================================
    CanonicalIntent.DOMAIN_DEBUG: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_DEBUG,
        description="Debug: investigate bugs, read logs, diagnose issues, fix code",
        behavior="Route to debug assistant with full tool access and project context",
        trigger_phrases=[
            "there's a bug",
            "debug this",
            "investigate that error",
            "check the logs",
            "something's broken",
            "fix that bug",
            "why is it crashing",
            "look at the error",
            "what went wrong",
            "diagnose the issue",
            "go to debug",
            "open debug",
            "show debug",
            "move to debug",
            "switch to debug",
            "debug tab",
            "it didn't work",
            "that didn't work",
            "why didn't that work",
            "why did that fail",
            "something went wrong",
            "it's not working",
            "that's not working",
            "the upload failed",
            "the video didn't work",
            "check the logs",
            "look at the logs",
            "what went wrong with",
            "why is it broken",
            "can you check why",
        ],
        trigger_patterns=[
            r"\b(bug|crash|error|exception|broken|glitch|issue|problem|fail)\b.*\b(fix|debug|investigate|diagnose|check|look)\b",
            r"\b(debug|diagnose|investigate|triage)\s+(this|that|the|it)\b",
            r"\b(didn'?t|did\s*not|doesn'?t|does\s*not)\s+(work|upload|load|connect|save|send|play)\b",
            r"\bwhy\s+(did|didn'?t|is|isn'?t)\s+(it|that|the|this)\s+(fail|work|break|crash|error)\b",
            r"\b(check|look\s+at|read)\s+(the\s+)?logs?\b",
            r"\b(check|read|show)\s+(the\s+)?(log|error|traceback|stack\s*trace)\b",
            r"\bwhy\s+(is|does|did)\s+(it|this|that)\s+(crash|fail|break|error)\b",
        ],
    ),

    # =========================================================================
    # EDUCATION
    # =========================================================================
    CanonicalIntent.DOMAIN_EDUCATION: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_EDUCATION,
        description="Education: study plans, learning progress, courses, philosophy",
        behavior="Pull education data from education service, inject into LLM context",
        trigger_phrases=[
            "my study plan",
            "learning progress",
            "what should I study",
            "education goals",
            "philosophy studies",
            "course progress",
            "what's next to learn",
            "go to education",
            "open education",
            "show education",
            "move to education",
            "switch to education",
            "education tab",
        ],
        trigger_patterns=[
            r"\b(study|learn|course|education|curriculum|lesson)\s*(plan|progress|goal|status)\b",
            r"\b(philosophy|psychology|economics|sociology|stoic|plato|aristotle|nietzsche|kant|marx)\b",
            r"\bwhat\s+(should|do)\s+I\s+(study|learn|read)\b",
            r"\b(AI and society|automation|UBI|universal basic income)\b",
        ],
    ),

    # =========================================================================
    # BUILDS
    # =========================================================================
    CanonicalIntent.DOMAIN_BUILDS: IntentDefinition(
        intent=CanonicalIntent.DOMAIN_BUILDS,
        description="Build pipeline: project build status, history, debug projects",
        behavior="Pull build data from builds service, inject into LLM context",
        trigger_phrases=[
            "build status",
            "last build",
            "build history",
            "project builds",
            "what was the last build",
            "how did the build go",
            "build report",
            "pipeline status",
            "go to builds",
            "open builds",
            "show builds",
            "move to builds",
            "switch to builds",
            "builds tab",
            "go to project builds",
            "open project builds",
            # Plain-text project modification phrases (substring match)
            "new feature for",
            "modify the app",
            "update the app",
            "extend the app",
            "change the app",
            "work on the app",
            "driver copilot",
            "driver co-pilot",
            "astra bridge",
        ],
        trigger_patterns=[
            r"\b(build|pipeline)\s*(status|history|report|result|log)\b",
            r"\bhow\s+did\s+the\s+(build|pipeline)\s+(go|do|finish|complete)\b",
            r"\blast\s+(build|pipeline|deploy)\b",
            # "add X to the app" — add before target
            r"\badd\b.*\b(element|feature|screen|module|component|page|tab)s?\b.*\b(app|copilot|co-?pilot|android|driver|bridge)\b",
            # "features to add to the app" — target before add (reverse order)
            r"\b(feature|element|screen|module|component|page|tab)s?\b.*\b(add|implement|build|create|include)\b.*\b(app|copilot|co-?pilot|android|driver|bridge)\b",
            # "features I want added to X" / "features for the app"
            r"\b(feature|element|screen|module|component)s?\b.*\b(for|into|to|in)\s+(the\s+)?(app|copilot|co-?pilot|android|driver|bridge)\b",
            # "app ... add ... feature" (app mentioned first)
            r"\b(app|copilot|co-?pilot|android|driver|bridge)\b.*\badd\b.*\b(element|feature|screen|module)s?\b",
            # "work on X"
            r"\bwork\s+on\b.*\b(copilot|co-?pilot|android|driver|app|bridge)\b",
            # "add X to Y" with regex (moved from trigger_phrases)
            r"\badd\s+.*\s+to\s+.*\b(copilot|co-?pilot|app|android|bridge)\b",
            r"\b(driver|android)\s*(co-?pilot)\b",
            # AstraBridge / Astra Bridge as a project name
            r"\b(astra\s*bridge)\b",
        ],
    ),
}

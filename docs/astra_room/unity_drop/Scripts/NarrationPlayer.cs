// FILE: Assets/AstraRoom/Scripts/NarrationPlayer.cs
// Purpose: Queued narration — POSTs text to the backend /scene/narrate (Chatterbox MP3),
//          plays clips through one AudioSource so narrations never overlap.
// Called-by: TimelineRunner (Enqueue per narrate event)
// Depends-on: AstraClient (HttpBaseUrl), UnityWebRequest + DownloadHandlerAudioClip
// Last-renovated: 2026-06-12

using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace AstraRoom
{
    [RequireComponent(typeof(AudioSource))]
    public class NarrationPlayer : MonoBehaviour
    {
        private readonly Queue<string> _queue = new Queue<string>();
        private AudioSource _audioSource;
        private Coroutine _worker;

        private void Awake()
        {
            _audioSource = GetComponent<AudioSource>();
            _audioSource.playOnAwake = false;
            _audioSource.spatialBlend = 0f; // narration is 2D voice-of-god, not positional
        }

        /// <summary>Queue one narration line; lines play strictly in order.</summary>
        public void Enqueue(string text)
        {
            _queue.Enqueue(text);
            if (_worker == null) _worker = StartCoroutine(Worker());
        }

        /// <summary>Drop pending lines and stop current audio (scene teardown).</summary>
        public void ClearQueue()
        {
            _queue.Clear();
            if (_worker != null) { StopCoroutine(_worker); _worker = null; }
            if (_audioSource != null && _audioSource.isPlaying) _audioSource.Stop();
        }

        private IEnumerator Worker()
        {
            while (_queue.Count > 0)
            {
                var text = _queue.Dequeue();
                yield return SynthesiseAndPlay(text);
            }
            _worker = null;
        }

        private IEnumerator SynthesiseAndPlay(string text)
        {
            var baseUrl = AstraClient.Instance != null
                ? AstraClient.Instance.HttpBaseUrl
                : "http://127.0.0.1:8000";
            var url = $"{baseUrl}/scene/narrate";
            var body = Encoding.UTF8.GetBytes(JsonUtility.ToJson(new NarrateBody { text = text }));

            using (var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
            {
                req.uploadHandler = new UploadHandlerRaw(body);
                req.downloadHandler = new DownloadHandlerAudioClip(url, AudioType.MPEG);
                req.SetRequestHeader("Content-Type", "application/json");
                req.timeout = 120; // Chatterbox synthesis can take a while on long lines

                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    // 503 = Chatterbox down — narration silently skipped, scene continues.
                    Debug.LogWarning($"[NarrationPlayer] narrate failed ({req.responseCode}): {req.error}");
                    AstraClient.Instance?.SendStatus("error", null,
                        $"narration failed ({req.responseCode}) — is Chatterbox running on :8002?");
                    yield break;
                }

                var clip = DownloadHandlerAudioClip.GetContent(req);
                if (clip == null)
                {
                    Debug.LogWarning("[NarrationPlayer] could not decode narration MP3");
                    yield break;
                }
                _audioSource.clip = clip;
                _audioSource.Play();
                // Wait for the line to finish so queued narrations never overlap.
                while (_audioSource.isPlaying) yield return null;
            }
        }

        [System.Serializable]
        private class NarrateBody
        {
            public string text;
        }
    }
}

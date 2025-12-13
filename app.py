import streamlit as st
import requests
import tweepy
import pandas as pd
import plotly.express as px

# ==============================================================================
# 🔑 PART 1: SECRETS AND CONFIGURATION
# ==============================================================================

# --- Your EmoSense API URL ---
EMOSENSE_API_URL = "https://srkr6115-emosense.hf.space/analyze"

# --- Your X API Bearer Token ---
X_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAEWN4wEAAAAAAO1IZ3nE3S1RwFqlPtmf4Krd%2F0o%3D6TSvCGYcScPjdCFD8rpvY82jRHIP7InD6S1tdzC6VHtYq6YbPq"

# --- Emoji dictionary for your emotions ---
EMOJI_MAP = {
    "anger": "😠",
    "annoyance": "😒",
    "approval": "👍",
    "caring": "❤️",
    "curiosity": "🤔",
    "fear": "😨",
    "gratitude": "🙏",
    "joy": "😄",
    "love": "🥰",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😮"
}


# ==============================================================================
# ⚙ PART 2: HELPER FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=900)  # Caches data for 15 minutes
def get_recent_tweets(query, bearer_token, tweet_count=10):
    """
    Fetches recent tweets for a given query using the X v2 API.
    Default set to 10 to save API quota.
    """
    try:
        client = tweepy.Client(bearer_token)
        # X API minimum max_results is 10
        response = client.search_recent_tweets(
            query=f"{query} -is:retweet lang:en",
            max_results=tweet_count
        )
        if response.data:
            return [tweet.text for tweet in response.data]
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return []

def analyze_emotion(text):
    """Sends text to your deployed EmoSense API."""
    try:
        response = requests.post(EMOSENSE_API_URL, json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.text
            return {"error": f"API returned status {response.status_code}: {error_detail}"}
    except requests.exceptions.RequestException:
        return {"error": f"Could not connect to EmoSense API at {EMOSENSE_API_URL}. Is your backend running?"}

# ==============================================================================
# 🖥 PART 3: THE FINAL, UPGRADED STREAMLIT UI
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="EmoSense: The Unified Emotion Monitor",
    page_icon="🧠",
    layout="wide"
)

# --- Title and Header ---
st.title("🧠 EmoSense: The Unified Emotion Monitor")
st.markdown("Analyzing real-time X data for Emotion, Intensity, *and* Sarcasm.")

# --- Test a Single Tweet Section ---
with st.expander("🔬 Test a Single Tweet", expanded=True): 
    single_tweet_text = st.text_input("Enter your own tweet to analyze (e.g., 'Oh great, another meeting')")
    
    if st.button("Analyze Single Tweet"):
        if single_tweet_text:
            with st.spinner("Analyzing your tweet..."):
                result = analyze_emotion(single_tweet_text)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # 1. Extract all data
                    emotions = result['emotions_confidence']
                    intensities = result['predicted_intensity']
                    sarcasm = result['sarcasm_score']
                    
                    # 2. Find the top emotion and its details
                    top_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[top_emotion]
                    intensity = intensities[top_emotion]
                    emoji = EMOJI_MAP.get(top_emotion, "🧠") 
                    
                    # 3. Create the 3-column metrics layout
                    st.markdown(f"**Analysis for:** *{single_tweet_text}*")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Top Emotion", f"{emoji} {top_emotion.capitalize()}", f"{confidence:.1%}")
                    col2.metric(f"{top_emotion.capitalize()} Intensity", f"{intensity:.2f} / 1.0")
                    col3.metric("Sarcasm Score", f"{sarcasm:.1%}")

                    # 4. Create the full emotion confidence chart
                    df_emotions = pd.DataFrame(emotions.items(), columns=['Emotion', 'Confidence'])
                    
                    fig = px.bar(df_emotions, 
                                 x='Confidence', 
                                 y='Emotion', 
                                 orientation='h',
                                 title="Full Emotion Analysis",
                                 color='Confidence', 
                                 color_continuous_scale='Blues' 
                    )
                    fig.update_layout(xaxis_title="Confidence Score (%)", yaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)

                    # 5. Keep the raw JSON in an expander
                    with st.expander("Show Full Analysis (JSON)"):
                        st.json(result)
        else:
            st.warning("Please enter some text to analyze.")

st.divider()

# --- Main "Topic Analysis" Feature ---
st.header("Live Topic Analysis")
query = st.text_input("Enter a topic or keyword to analyze:", key="query_input", placeholder="e.g., iPhone 17")

if st.button("Analyze Topic ✨"):
    if not query:
        st.warning("Please enter a topic to analyze.")
    else:
        # UPDATED: Spinner text now says "10 live tweets"
        with st.spinner(f"Fetching up to 10 live tweets for '{query}' and analyzing all 3 tasks..."):
            
            # 1. Fetch tweets (Defaults to 10 now)
            tweets = get_recent_tweets(query, X_BEARER_TOKEN, tweet_count=10) 

            if not tweets:
                st.error(f"No recent tweets found for '{query}'. Please try another topic or wait for your API quota to reset.")
            else:
                # 2. Analyze tweets
                results = [analyze_emotion(tweet) for tweet in tweets]
                valid_results = [r for r in results if "error" not in r and r.get('text')]

                if not valid_results:
                    st.error("Analysis failed for all fetched tweets. Please check the EmoSense API connection.")
                else:
                    st.success(f"Analyzed {len(valid_results)} tweets successfully!")
                    
                    # --- 3. Extract data for all 3 tasks ---
                    df_emotions = pd.DataFrame([res['emotions_confidence'] for res in valid_results])
                    df_intensity = pd.DataFrame([res['predicted_intensity'] for res in valid_results])
                    sarcasm_scores = [res['sarcasm_score'] for res in valid_results]

                    # 4. Aggregate all 3 metrics ---
                    avg_emotions = df_emotions.mean()
                    avg_intensity = df_intensity.mean()
                    avg_sarcasm = pd.Series(sarcasm_scores).mean()
                    
                    st.subheader(f"Overall Analysis for '{query}'")
                    
                    # --- 5. Create a 2-column layout for key metrics ---
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Key Metrics")
                        top_emotion = avg_emotions.idxmax()
                        st.metric(label="Most Dominant Emotion", 
                                  value=f"{EMOJI_MAP.get(top_emotion, '🧠')} {top_emotion.capitalize()}", 
                                  delta=f"{avg_emotions.max():.1%} confidence")
                        
                        st.metric(label="Average Sarcasm Score", 
                                  value=f"{avg_sarcasm:.1%}") 
                        st.info("A high sarcasm score means the literal emotions (like 'joy') should be treated with caution.")

                    with col2:
                        st.markdown("#### Average Emotion Intensity")
                        fig_intensity = px.bar(avg_intensity, x=avg_intensity.values, y=avg_intensity.index, orientation='h',
                                               labels={'x': 'Average Intensity Score (0-1)', 'y': 'Emotion'})
                        fig_intensity.update_layout(xaxis_range=[0,1], yaxis_title=None, coloraxis_showscale=False)
                        st.plotly_chart(fig_intensity, use_container_width=True)

                    # --- 6. Show the original emotion confidence chart ---
                    st.markdown("#### Average Emotion Confidence")
                    fig_emotion = px.bar(avg_emotions, x=avg_emotions.values, y=avg_emotions.index, orientation='h',
                                         labels={'x': 'Average Confidence Score (%)', 'y': 'Emotion'})
                    fig_emotion.update_layout(yaxis_title=None, coloraxis_showscale=False)
                    st.plotly_chart(fig_emotion, use_container_width=True)

                    st.divider()

                    # --- 7. Upgraded Individual Tweet Analysis ---
                    st.subheader("Analysis of Individual Tweets")
                    st.markdown("This shows the full power of the unified model on each tweet.")
                    
                    for res in valid_results: # Show all 10 results since list is small
                        with st.container(border=True):
                            st.write(f"*Tweet:* {res['text']}")
                            
                            # Get all 3 metrics
                            top_emotion = max(res['emotions_confidence'], key=res['emotions_confidence'].get)
                            confidence = res['emotions_confidence'][top_emotion]
                            intensity = res['predicted_intensity'][top_emotion]
                            sarcasm = res['sarcasm_score']
                            emoji = EMOJI_MAP.get(top_emotion, "🧠")
                            
                            # Display all 3 metrics
                            t_col1, t_col2, t_col3 = st.columns(3)
                            t_col1.metric("Top Emotion", f"{emoji} {top_emotion.capitalize()}", f"{confidence:.1%}")
                            t_col2.metric(f"{top_emotion.capitalize()} Intensity", f"{intensity:.2f}")
                            t_col3.metric("Sarcasm", f"{sarcasm:.1%}")
                            
                            with st.expander("Show Full Analysis (JSON)"):
                                st.json(res)
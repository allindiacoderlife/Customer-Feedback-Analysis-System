// Predictor Page JavaScript with GSAP Animations

// Sample feedback texts for testing
const sampleFeedbacks = [
  "The product quality is excellent and delivery was fast. Highly recommend!",
  "Terrible experience. The item arrived damaged and customer service was unhelpful.",
  "Product is okay. Nothing special but does what it's supposed to do.",
  "Amazing customer support! They resolved my issue within minutes. Very impressed!",
  "Disappointed with the quality. Not worth the price at all.",
  "Good value for money. Decent product with reasonable shipping time.",
  "The worst purchase I've made. Complete waste of money and time.",
  "Fantastic! Exceeded all my expectations. Will definitely buy again!",
  "Average product. Some features work well, others could be improved.",
  "Outstanding quality and excellent packaging. Very happy with this purchase!"
];

// Page load animations
document.addEventListener('DOMContentLoaded', () => {
  // Animate page header
  gsap.from('.page-header', {
    duration: 0.8,
    y: -30,
    opacity: 0,
    ease: 'power3.out'
  });

  // Animate predictor card
  gsap.from('.predictor-card', {
    duration: 1,
    y: 50,
    opacity: 0,
    ease: 'power3.out',
    delay: 0.2
  });

  // Animate tips section
  gsap.from('.tips-section', {
    duration: 1,
    y: 50,
    opacity: 0,
    ease: 'power3.out',
    delay: 0.4
  });

  // Stagger animate tip cards
  gsap.from('.tip-card', {
    duration: 0.6,
    y: 30,
    opacity: 0,
    stagger: 0.1,
    ease: 'power2.out',
    delay: 0.6
  });

  // Character counter
  const textarea = document.getElementById('feedback-text');
  const charCount = document.getElementById('char-count');

  textarea.addEventListener('input', () => {
    const count = textarea.value.length;
    charCount.textContent = count;

    // Animate counter on change
    gsap.from(charCount, {
      duration: 0.3,
      scale: 1.3,
      ease: 'back.out'
    });
  });
});

// Load sample feedback
function loadSample() {
  const randomIndex = Math.floor(Math.random() * sampleFeedbacks.length);
  const textarea = document.getElementById('feedback-text');
  const charCount = document.getElementById('char-count');
  const sample = sampleFeedbacks[randomIndex];

  // Animate text input
  gsap.to(textarea, {
    duration: 0.3,
    opacity: 0,
    onComplete: () => {
      textarea.value = sample;
      charCount.textContent = sample.length;
      
      // Animate back to visible
      gsap.to(textarea, {
        duration: 0.3,
        opacity: 1
      });
    }
  });
}

// Clear input
function clearInput() {
  const textarea = document.getElementById('feedback-text');
  const charCount = document.getElementById('char-count');

  // Animate clear
  gsap.to(textarea, {
    duration: 0.3,
    opacity: 0,
    onComplete: () => {
      textarea.value = '';
      charCount.textContent = '0';
      gsap.to(textarea, {
        duration: 0.3,
        opacity: 1
      });
    }
  });

  // Hide result, show empty state
  const emptyState = document.getElementById('prediction-empty');
  const resultState = document.getElementById('prediction-result');

  gsap.to(resultState, {
    duration: 0.4,
    opacity: 0,
    y: 20,
    onComplete: () => {
      resultState.classList.add('hidden');
      emptyState.classList.remove('hidden');
      gsap.from(emptyState, {
        duration: 0.4,
        opacity: 0,
        y: -20
      });
    }
  });
}

// Analyze sentiment
async function analyzeSentiment() {
  const textarea = document.getElementById('feedback-text');
  const text = textarea.value.trim();

  if (!text) {
    // Shake animation for empty input
    gsap.to(textarea, {
      duration: 0.1,
      x: -10,
      yoyo: true,
      repeat: 5,
      ease: 'power1.inOut'
    });
    alert('Please enter some feedback text to analyze.');
    return;
  }

  const analyzeBtn = document.getElementById('analyze-btn');
  const originalText = analyzeBtn.innerHTML;

  // Disable button and show loading
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing...';

  // Animate button
  gsap.to(analyzeBtn, {
    duration: 0.3,
    scale: 0.95
  });

  try {
    const startTime = Date.now();

    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('API Error:', errorData);
      throw new Error(errorData.error || 'Failed to get prediction');
    }

    const data = await response.json();
    console.log('Prediction data:', data); // Debug log
    
    const endTime = Date.now();
    const analysisTime = endTime - startTime;

    // Display results with animation
    displayResult(data, analysisTime);
  } catch (error) {
    console.error('Error:', error);
    alert('Error analyzing sentiment: ' + error.message);
  } finally {
    // Re-enable button
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = originalText;
    gsap.to(analyzeBtn, {
      duration: 0.3,
      scale: 1
    });
  }
}

// Display prediction result with animations
function displayResult(data, analysisTime) {
  console.log('Displaying result:', data); // Debug log
  
  const emptyState = document.getElementById('prediction-empty');
  const resultState = document.getElementById('prediction-result');

  // Hide empty state
  if (!emptyState.classList.contains('hidden')) {
    gsap.to(emptyState, {
      duration: 0.3,
      opacity: 0,
      y: -20,
      onComplete: () => {
        emptyState.classList.add('hidden');
      }
    });
  }

  // Update result values - API returns 'prediction' not 'sentiment'
  const sentiment = data.prediction || data.sentiment;
  const confidence = Math.round(data.confidence * 100);
  const probabilities = data.probabilities;

  console.log('Sentiment:', sentiment, 'Confidence:', confidence, 'Probabilities:', probabilities); // Debug

  // Capitalize first letter for display
  const sentimentDisplay = sentiment.charAt(0).toUpperCase() + sentiment.slice(1);
  
  document.getElementById('sentiment-value').textContent = sentimentDisplay;
  document.getElementById('sentiment-value').className = `sentiment-badge ${sentiment.toLowerCase()}-badge`;
  document.getElementById('analysis-time').textContent = `Analyzed in ${analysisTime}ms`;

  // Animate confidence bar
  const confidenceFill = document.getElementById('confidence-fill');
  confidenceFill.className = `confidence-fill ${sentiment.toLowerCase()}-fill`;

  gsap.to(confidenceFill, {
    duration: 1,
    width: `${confidence}%`,
    ease: 'power2.out'
  });

  // Update confidence text immediately (no animation to avoid NaN)
  document.getElementById('confidence-text').textContent = `${confidence}%`;

  // Update and animate probabilities - handle both lowercase and capitalized keys
  updateProbability('positive', probabilities.positive || probabilities.Positive || 0);
  updateProbability('negative', probabilities.negative || probabilities.Negative || 0);
  updateProbability('neutral', probabilities.neutral || probabilities.Neutral || 0);

  // Show result with animation
  resultState.classList.remove('hidden');
  gsap.from(resultState, {
    duration: 0.6,
    opacity: 0,
    y: 30,
    ease: 'power3.out'
  });

  // Stagger animate probability items
  gsap.from('.prob-item', {
    duration: 0.5,
    x: -20,
    opacity: 0,
    stagger: 0.1,
    ease: 'power2.out',
    delay: 0.3
  });
}

// Update probability with animation
function updateProbability(type, value) {
  const percentage = Math.round(value * 100);
  const barElement = document.getElementById(`bar-${type}`);
  const textElement = document.getElementById(`prob-${type}`);

  // Animate bar width
  gsap.to(barElement, {
    duration: 1,
    width: `${percentage}%`,
    ease: 'power2.out',
    delay: 0.2
  });

  // Update text value immediately (no animation to avoid NaN)
  if (textElement) {
    textElement.textContent = `${percentage}%`;
  }
}

// Add hover animations to buttons
document.querySelectorAll('.btn-primary, .btn-secondary, .btn-outline').forEach(btn => {
  btn.addEventListener('mouseenter', () => {
    gsap.to(btn, {
      duration: 0.3,
      scale: 1.05,
      ease: 'power2.out'
    });
  });

  btn.addEventListener('mouseleave', () => {
    gsap.to(btn, {
      duration: 0.3,
      scale: 1,
      ease: 'power2.out'
    });
  });
});

// Enter key to analyze
document.getElementById('feedback-text').addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') {
    analyzeSentiment();
  }
});

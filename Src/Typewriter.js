document.addEventListener("DOMContentLoaded", function () {
    const phrases = [
        "Personalized Career Roadmaps.",
        "Skill-Based Role Recommendations.",
        "Live Job Market Insight.",
        "Domain-Specific Discussion Channels."
    ];

    let currentPhraseIndex = 0;
    let currentCharacterIndex = 0;
    const typingSpeed = 100; 
    const erasingSpeed = 50; 
    const delayBetweenPhrases = 1500; 

    const targetElement = document.querySelector("#animated-text");

    function typeWriterEffect() {
        if (currentCharacterIndex < phrases[currentPhraseIndex].length) {
            targetElement.textContent += phrases[currentPhraseIndex][currentCharacterIndex];
            currentCharacterIndex++;
            setTimeout(typeWriterEffect, typingSpeed);
        } else {
            setTimeout(eraseText, delayBetweenPhrases);
        }
    }

    function eraseText() {
        if (currentCharacterIndex > 0) {
            targetElement.textContent = phrases[currentPhraseIndex].substring(0, currentCharacterIndex - 1);
            currentCharacterIndex--;
            setTimeout(eraseText, erasingSpeed);
        } else {
            currentPhraseIndex = (currentPhraseIndex + 1) % phrases.length;
            setTimeout(typeWriterEffect, typingSpeed);
        }
    }

    typeWriterEffect();
});

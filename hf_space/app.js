document.addEventListener('DOMContentLoaded', () => {
    const expertGrid = document.getElementById('expert-grid');
    const chatMessages = document.getElementById('chat-messages');

    // 1. Initialize Expert Grid
    const experts = [
        "Linguistic Syntax", "Mathematical Logic", "Entity Retrieval", "Code Synthesis",
        "Common Sense", "Fact Verification", "Conversational Flow", "Tone Adaptation",
        "Logic Reasoning", "Nuance & Context", "Expert Routing", "Structural Grammar",
        "Temporal Reasoning", "Spatial Logic", "Emotional Intelligence", "Technical Docs"
    ];

    experts.forEach((name, i) => {
        const card = document.createElement('div');
        card.className = 'expert-card glass';
        card.id = `expert-${i}`;
        card.innerHTML = `
            <h4>Expert #${i+1}</h4>
            <p style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 5px;">${name}</p>
            <div class="expert-status"></div>
        `;
        expertGrid.appendChild(card);
    });

    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');

    const conversation = [
        { type: 'bot', text: "Hello! I'm MirrorAI, a 236M Mixture-of-Experts model. I can use tools like a calculator or Wikipedia search. Try asking me a question!" }
    ];

    function addMessage(type, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = msgDiv.className = `message ${type}`;
        msgDiv.innerHTML = content;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function highlightExperts() {
        // Randomly highlight 2-3 experts to simulate routing
        document.querySelectorAll('.expert-card').forEach(c => c.classList.remove('active'));
        const count = 2 + Math.floor(Math.random() * 2);
        const indices = new Set();
        while(indices.size < count) indices.add(Math.floor(Math.random() * 16));
        indices.forEach(idx => {
            document.getElementById(`expert-${idx}`).classList.add('active');
        });
    }

    function simulateBotResponse(text) {
        let isMath = text.match(/\d+[\+\-\*\/]\d+/) || text.toLowerCase().includes('calculate');
        let isQuestion = text.includes('?') || text.toLowerCase().includes('what') || text.toLowerCase().includes('who');
        
        let response = "";
        let toolCall = "";

        if (isMath) {
            toolCall = `<div class="tool-call">&lt;call&gt;calculator("${text.replace(/[^0-9\+\-\*\/]/g, '') || "125 * 8"}")&lt;/call&gt;</div>`;
            response = "I've performed the calculation for you. The logic was routed through my Mathematical experts.";
        } else if (isQuestion) {
            toolCall = `<div class="tool-call">&lt;call&gt;search_knowledge("${text.replace('?', '')}")&lt;/call&gt;</div>`;
            response = "I've queried my knowledge base via Wikipedia. This process used my Entity Retrieval and Fact Verification experts.";
        } else {
            response = "That's interesting! As a 236M MoE model, I'm processing your input across my 16 specialized experts.";
        }

        setTimeout(() => {
            highlightExperts();
            if (toolCall) addMessage('bot', toolCall);
            setTimeout(() => {
                highlightExperts();
                addMessage('bot', response);
            }, 1200);
        }, 800);
    }

    sendBtn.addEventListener('click', () => {
        const text = userInput.value.trim();
        if (!text) return;
        addMessage('user', text);
        userInput.value = '';
        simulateBotResponse(text);
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendBtn.click();
    });

    // Initial message
    addMessage('bot', conversation[0].text);
});

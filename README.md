# Systems Analysis and Design: Team 14
**Competition:** Psychopathy Prediction on Twitter  

---

## Table of Contents
- [Team Members](#team-members)
- [Workshop #1](#workshop-1-summary)
- [Workshop #2](#workshop-2-summary)

---

## Team Members
- Juan David Ávila 20232020154
- Raúl Andres Diaz 20232020058
- Juan David Bejarano Cristancho 20232020056
- David Sanchez Acero 20232020049

---

## Workshop #1 Summary
In this workshop, we developed the **first phase of the project**, which included: 

- **Competition review:** Understanding the *Psychopathy Prediction Based on Twitter Usage* challenge organized by the Online Privacy Foundation (2012).  
- **Preliminary problem analysis:** Identifying the main objectives (predicting psychopathy in Twitter users) and understanding the evaluation metrics.  
- **Initial data exploration:** Reviewing the dataset characteristics (≈2,927 users, over 3 million tweets, 337 derived variables).  
- **Work plan design:** Defining tasks and team roles for data collection, feature engineering, and preparation for machine learning models.  
- **Complexity discussion:** Identifying challenges such as extreme class imbalance, high dimensionality, and sensitivity to initial conditions.
- [Click here to view the Workshop #1 PDF](Docs/Workshop_1.pdf)

---

## Workshop #2 Summary
- During this second workshop, the team moved from the analytical phase carried out in Workshop #1 to the **design stage** of the psychopathy prediction system. The goal was to translate the previous findings—related to complexity, chaos, and sensitivity—into a clear and structured system proposal.
- Throughout the session, the group designed a **modular system architecture** capable of handling large volumes of linguistic and behavioral data from Twitter. This included defining how data would be collected, processed, and analyzed through different layers such as data ingestion, feature engineering, modeling, and evaluation.
- Based on the challenges identified earlier, several design decisions were made to **control sensitivity and chaotic behavior**, including organizing the workflow to improve reproducibility and integrating mechanisms for feedback and monitoring during model development.
- The team also prepared an **implementation plan** that outlines the main stages of the project—from data preparation to deployment—and selected the tools that will support this process, such as Python libraries for analysis and GitHub for collaboration.
- [Click here to view the Workshop #2 PDF](Docs/Workshop_2.pdf)

---

## Workshop #3 Summary
In this third workshop, the team focused on refining the **system architecture** and establishing a **robust project management plan** to ensure reliability, ethical operation, and continuous improvement of the *Psychopathy Prediction System*.

- The group restructured the system into a **layered, feedback-driven architecture** that transforms unstructured Twitter data into interpretable insights. This new design includes components for data cleaning, behavioral feature analysis, prediction engines, ethical control, and system monitoring.  
- Each layer was analyzed in detail to understand its function and interconnection, emphasizing **stability, fairness, and modularity**. The team also introduced an **Ethics & Control Layer** to guarantee responsible use of data and prevent discriminatory outcomes.  
- A **Quality and Risk Analysis** was conducted to identify potential issues such as data imbalance, feedback instability, and latency. Specific mitigation strategies were proposed, including resampling methods, performance thresholds, and monitoring mechanisms.  
- The **Project Management Plan** defined clear roles for each team member, a Kanban-based workflow using Trello, and a timeline of milestones for model development, evaluation, and integration.  
- Additionally, several **incremental improvements** were made, such as establishing controlled feedback loops, integrating a shared Data Hub, clarifying task priorities, and adding parallel layers for ethics and monitoring.  
- Finally, the team reflected on how robustness in system design arises not from complexity but from awareness and adaptability. The updated architecture now behaves as a **self-regulating structure**, capable of maintaining stability and ethical integrity even in dynamic environments.  
[Click here to view the Workshop #3 PDF](Docs/Workshop_3.pdf)

---

## Workshop #4 summary
In this fourth workshop, the team transitioned from system design and architectural refinement toward the implementation and simulation phase of the Psychopathy Prediction System.Building on the insights developed in Workshops #1, #2, and #3, the objective of this stage was to validate the proposed architecture through computational experimentation and to explore emergent behaviors within the system.

-The group implemented two complementary simulation approaches designed to test different dimensions of the system. 
-The first focused on data-driven simulation, using classical machine learning models to mimic basic processes such as training, evaluation, and prediction. This scenario allowed the team to observe learning dynamics, examine performance variations, and detect potential sensitivity issues in the model process. 
-The second simulation was event-based and used an adapted simulation model to represent spatial or event-driven interactions relevant to the system's behavioral analysis components. The simulations focused on predefined architectural layers, ensuring consistency with the system's modular workflow.
-Throughout the workshop, the team ran these simulations with different configurations to analyze how changes in parameters, perturbations, or data segments affected the system.
-This experimentation identified bottlenecks, emerging patterns, and potential chaotic behaviors, reinforcing the importance of feedback control and stable workflow design.

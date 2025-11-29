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
In this fourth workshop, the team completed the practical implementation and simulation-based analysis of the Psychopathy Prediction System, translating architectural design from Workshop 3 into a working machine learning pipeline.

- The group executed a complete data balancing strategy using SMOGN and DOOM Data techniques to address severe class imbalance (only ~3% in high psychopathy range), enabling effective learning of extreme cases.
- A Random Forest Regressor model was trained on balanced data, achieving exceptional precision with test MSE of 0.000070. Model stability was validated across multiple seeds (MSE range: 0.000031–0.000219).
- Stability and Sensitivity Analysis revealed robustness to small perturbations (1–3% noise) but fragility under larger disturbances (10% noise: MSE 0.003689), establishing critical requirements for input validation in production.
- Event-Based Simulation employed a cellular automaton to model emergent behavioral dynamics, tracking how psychopathy traits propagate through populations via neighborhood influence and temporal evolution patterns.
- Limitations were documented with a comprehensive roadmap for future work, including percentile-specific metrics, robust ensemble models (XGBoost, LightGBM), and continuous validation pipelines.
- Workshop 4 demonstrated how theoretical architecture principles translate into measurable system performance, establishing a foundation for production deployment with clear understanding of model capabilities and operational safeguards.
[Click here to view the Workshop #4 PDF](Docs/Workshop_4.pdf)

Relative estimation is a concept that simply means "comparing two things to each other." If you've ever said something like, "Hey, this tree is twice as tall as that tree," you already know how to do it.

In the world of software development and data science projects, estimating work has always been challenging. Traditional approaches often fall short because they try to predict absolute time requirements, which can be highly variable depending on countless factors. Relative estimation offers a more pragmatic and often more accurate approach.

## Agile Estimation

**Agile estimation** is intended to capture:

1. The *amount* of work.
2. The *difficulty* of the work.
3. The *risk inherent* in the work.

When paired with relative estimation, it results in questions like, "Is this feature as complicated as the other feature we built last week?" Or, "If we take on this feature, is it riskier than the other feature?".

When work items are similar enough across these questions, we give them the same number on a scale (see table below) that the team has previously selected. As work items differ, the team discusses those differences to understand just how different the work is, relatively, and gives a corresponding number on that same scale.

![](/img/jira-story-estimation/jira-inline.png)

Amongst these, the most popular is the *Modified Fibonacci* scale.

<aside>
💡 It was documented in a fantastic book titled Agile Estimating and Planning by Mike Cohn.

</aside>

Its popularity is for a good reason:

The nonlinear sequence (of the Fibonacci numbers) works well because the gaps reflect the greater levels of uncertainty associated with estimates for bigger, more unknown pieces of work.

Agile teams call these unitless measures of story size "story points," or simply "points." It is recommended that teams avoid the temptation to continue to estimate in hours, even when deliberately using relative scales because the perception by stakeholders is often a higher degree of certainty than there truly is.

> At one point, "ideal hours" were offered as an alternative to story points. An "ideal hour" is a mythical hour where you can work, uninterrupted, with all the knowledge and resources at your fingertips to get the job done. In practice, this causes too much confusion among the team and stakeholders—someone inevitably confuses an ideal hour estimate with an actual hour from reality.
>


## Estimation must be a Team Event

A generally accepted practice is to have the team make estimates together. This practice allows the team to discover how well work is understood, encourages knowledge sharing, builds team cohesiveness, and fosters buy-in. One of the most popular techniques for having these team conversations to obtain estimates is Planning Poker [Cohn 2016]. The gist of it is that the team talks about its work in a structured (but fun) way. The team estimates together, they drive conversation, and they expose uncertainty. And numbers pop out that are just good enough to move forward.

## Planning Poker Process

Planning Poker is a consensus-based, gamified technique for estimating effort. Here's how it works:

### Step-by-Step Process

1. **Present the Story**: The product owner or scrum master presents a user story to the team.

2. **Discussion Phase**: Team members ask questions and discuss:
   - Requirements clarification
   - Potential technical challenges
   - Dependencies and assumptions
   - Definition of done

3. **Individual Estimation**: Each team member privately selects a card representing their estimate (using the chosen scale, e.g., Modified Fibonacci).

4. **Reveal Cards**: All team members reveal their cards simultaneously.

5. **Discuss Differences**: If estimates vary significantly:
   - Those with highest and lowest estimates explain their reasoning
   - Team discusses different perspectives and approaches
   - Additional questions may be raised

6. **Re-estimate**: Team members select new cards based on the discussion.

7. **Converge**: Repeat steps 4-6 until the team reaches consensus or a reasonable range.

## Estimation Scales in Detail

### Modified Fibonacci (Most Popular)
**Scale**: 1, 2, 3, 5, 8, 13, 21, 40, 100, ∞, ?

**Rationale**: 
- Reflects the uncertainty that increases with size
- Forces teams to make meaningful distinctions between similar-sized items
- The gaps become larger for bigger items, acknowledging greater uncertainty

### T-Shirt Sizing
**Scale**: XS, S, M, L, XL, XXL

**Best for**:
- Initial rough sizing
- Non-technical stakeholders
- Portfolio-level planning

### Powers of 2
**Scale**: 1, 2, 4, 8, 16, 32

**Benefits**:
- Simple doubling relationship
- Easy to understand scaling
- Good for teams preferring simpler math

## Common Pitfalls and Solutions

### Pitfall 1: Converting Points to Hours
**Problem**: Teams try to create direct conversions (e.g., "1 point = 4 hours")
**Solution**: Focus on relative sizing and use velocity for planning

### Pitfall 2: Estimation Pressure
**Problem**: Management pressures for faster estimation or specific numbers
**Solution**: Educate stakeholders on the benefits of proper estimation and its impact on delivery predictability

### Pitfall 3: Individual Estimates
**Problem**: Having one person estimate instead of the team
**Solution**: Ensure the whole team participates in estimation sessions

### Pitfall 4: Perfectionist Estimation
**Problem**: Spending too much time trying to get "perfect" estimates
**Solution**: Remember that estimates are meant to be "good enough" for planning

## Benefits for Data Science Teams

Relative estimation is particularly valuable for data science teams because:

### Handling Uncertainty
- Data science work often involves exploration and experimentation
- Relative estimation acknowledges inherent uncertainty without false precision
- Easier to compare "research spike" tasks against known quantities

### Cross-functional Collaboration
- Enables collaboration between data scientists, engineers, and product managers
- Creates shared understanding of work complexity
- Facilitates better resource allocation

### Improved Planning
- Velocity becomes more predictable over time
- Better sprint planning and release forecasting
- Helps identify when work needs to be broken down further

## Best Practices

### For Effective Estimation Sessions
1. **Time-box discussions** (typically 5-10 minutes per story)
2. **Focus on relative size**, not absolute time
3. **Include the whole team** in estimation
4. **Use reference stories** as anchors for comparison
5. **Break down large items** (anything over 13 points)

### For Long-term Success
1. **Track velocity** over multiple sprints
2. **Review and retrospect** on estimation accuracy
3. **Refine your reference stories** based on completed work
4. **Educate stakeholders** on how to use estimates for planning

## Conclusion

Relative estimation, particularly when implemented through techniques like Planning Poker, transforms the estimation process from a dreaded individual activity to a collaborative team exercise that builds understanding and alignment. 

For data science teams working in agile environments, mastering relative estimation techniques is crucial for successful project delivery and stakeholder management. The key is to embrace the inherent uncertainty in our work while providing enough structure for effective planning and communication.

Remember: the goal isn't perfect estimates—it's estimates that are good enough to make informed decisions and improve over time through team learning and calibration.
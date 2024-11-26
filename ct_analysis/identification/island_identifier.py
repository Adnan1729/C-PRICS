# IslandIdentifier class

class IslandIdentifier:
    def __init__(self, distance_threshold=5.0, overlap_threshold=0.3):
        """
        Initialize IslandIdentifier with configurable thresholds
        
        Parameters:
        distance_threshold: Maximum distance (meters) between island boundaries for matching
        overlap_threshold: Minimum overlap ratio required for potential splits/merges
        """
        self.distance_threshold = distance_threshold
        self.overlap_threshold = overlap_threshold
        self.registered_islands = pd.DataFrame(columns=[
            'global_id',
            'start_location',
            'end_location',
            'first_date',
            'last_date',
            'status',  # 'active', 'merged', 'split', 'disappeared'
            'related_islands',  # IDs of split/merged islands
            'confidence'  # confidence score for location matching
        ])
        self.next_id = 1
        self.history = []  # Track all changes to islands

    def calculate_overlap(self, island1, island2):
        """Calculate overlap ratio between two islands"""
        start = max(island1['start_location'], island2['start_location'])
        end = min(island1['end_location'], island2['end_location'])
        
        if end <= start:
            return 0.0
            
        overlap = end - start
        length1 = island1['end_location'] - island1['start_location']
        length2 = island2['end_location'] - island2['start_location']
        
        return overlap / min(length1, length2)

    def find_matches(self, island, current_date):
        """
        Find potential matches for an island
        Returns matches with confidence scores
        """
        active_islands = self.registered_islands[
            self.registered_islands['status'] == 'active'
        ]
        
        matches = []
        for _, registered in active_islands.iterrows():
            # Calculate different matching metrics
            start_diff = abs(registered['start_location'] - island['start_location'])
            end_diff = abs(registered['end_location'] - island['end_location'])
            overlap = self.calculate_overlap(island, registered)
            
            # Check if it's a potential match
            if (start_diff <= self.distance_threshold and 
                end_diff <= self.distance_threshold):
                
                # Calculate confidence score (0-1)
                confidence = 1.0 - max(
                    start_diff / self.distance_threshold,
                    end_diff / self.distance_threshold
                ) * (1 - overlap)
                
                matches.append({
                    'global_id': registered['global_id'],
                    'confidence': confidence,
                    'overlap': overlap
                })
        
        return sorted(matches, key=lambda x: x['confidence'], reverse=True)

    def register_new_island(self, island, date):
        """Register a new island and return its global ID"""
        global_id = f"IS_{self.next_id:04d}"
        self.next_id += 1
        
        new_island = pd.DataFrame({
            'global_id': [global_id],
            'start_location': [island['start_location']],
            'end_location': [island['end_location']],
            'first_date': [date],
            'last_date': [date],
            'status': ['active'],
            'related_islands': [[]],
            'confidence': [1.0]
        })
        
        self.registered_islands = pd.concat([self.registered_islands, new_island], 
                                          ignore_index=True)
        
        self.history.append({
            'date': date,
            'event': 'new',
            'global_id': global_id,
            'location': f"{island['start_location']:.1f}-{island['end_location']:.1f}m"
        })
        
        return global_id

    def update_island(self, global_id, island, date, confidence):
        """Update an existing island's record"""
        idx = self.registered_islands['global_id'] == global_id
        self.registered_islands.loc[idx, 'last_date'] = date
        self.registered_islands.loc[idx, 'confidence'] = min(
            self.registered_islands.loc[idx, 'confidence'].iloc[0],
            confidence
        )

    def check_splits_and_merges(self, current_islands, date):
        """Check for potential split or merged islands"""
        active_islands = self.registered_islands[
            self.registered_islands['status'] == 'active'
        ]
        
        # Group current islands by proximity
        grouped_current = []
        for island in current_islands:
            added = False
            for group in grouped_current:
                if any(self.calculate_overlap(island, existing) > self.overlap_threshold 
                      for existing in group):
                    group.append(island)
                    added = True
                    break
            if not added:
                grouped_current.append([island])
        
        # Check each group for splits/merges
        for group in grouped_current:
            if len(group) > 1:  # Potential split
                matches = []
                for island in group:
                    matches.extend(self.find_matches(island, date))
                
                if len(set(m['global_id'] for m in matches)) == 1:
                    # Split detected
                    parent_id = matches[0]['global_id']
                    self.mark_split(parent_id, group, date)
            
            elif len(group) == 1 and len(self.find_matches(group[0], date)) > 1:
                # Potential merge
                self.mark_merge(group[0], self.find_matches(group[0], date), date)

    def mark_split(self, parent_id, new_islands, date):
        """Mark an island as split and create new islands"""
        # Update parent island
        idx = self.registered_islands['global_id'] == parent_id
        self.registered_islands.loc[idx, 'status'] = 'split'
        
        # Create new islands
        new_ids = []
        for island in new_islands:
            new_id = self.register_new_island(island, date)
            new_ids.append(new_id)
        
        self.history.append({
            'date': date,
            'event': 'split',
            'parent_id': parent_id,
            'child_ids': new_ids
        })

    def mark_merge(self, new_island, parent_matches, date):
        """Mark islands as merged and create a new merged island"""
        # Update parent islands
        parent_ids = [m['global_id'] for m in parent_matches]
        for parent_id in parent_ids:
            idx = self.registered_islands['global_id'] == parent_id
            self.registered_islands.loc[idx, 'status'] = 'merged'
        
        # Create new merged island
        new_id = self.register_new_island(new_island, date)
        
        self.history.append({
            'date': date,
            'event': 'merge',
            'parent_ids': parent_ids,
            'new_id': new_id
        })

    def register_islands(self, df_date, date):
        """
        Register islands from a specific date and assign consistent global IDs
        """
        df = df_date.copy()
        df['global_id'] = None
        df['match_confidence'] = 1.0
        
        # Process each island in the current date
        current_islands = df.to_dict('records')
        
        # Check for splits and merges
        self.check_splits_and_merges(current_islands, date)
        
        # Process each island
        for idx, island in df.iterrows():
            matches = self.find_matches(island, date)
            
            if not matches:
                # New island
                global_id = self.register_new_island(island, date)
                df.loc[idx, 'global_id'] = global_id
                df.loc[idx, 'match_confidence'] = 1.0
            else:
                # Use best match
                best_match = matches[0]
                df.loc[idx, 'global_id'] = best_match['global_id']
                df.loc[idx, 'match_confidence'] = best_match['confidence']
                self.update_island(best_match['global_id'], island, date, 
                                 best_match['confidence'])
        
        return df
    
    def get_history_summary(self):
        """Return a summary of island history"""
        return pd.DataFrame(self.history)
